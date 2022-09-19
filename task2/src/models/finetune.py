import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig,RobertaModel
from transformers import BertConfig,BertModel
import numpy as np
import sys
sys.path.append("..") 
from src.utils import FocalLoss
from transformers import AutoModel, AutoConfig

class FtModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.pretrain_model_path is not None:
            print(f"持续预训练模型路径:{args.pretrain_model_path}")
            if "roberta" in  args.bert_dir:
                self.config = RobertaConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = RobertaModel(self.config, add_pooling_layer=False) 
            else:
                self.config = BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = BertModel(self.config, add_pooling_layer=False) 
            
            ckpoint = torch.load(args.pretrain_model_path)
            self.roberta.load_state_dict(ckpoint["model_state_dict"])
        else:
            self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) 
        # self.att_head = AttentionHead(self.config.hidden_size * 4, self.config.hidden_size)
        if args.premise:
            args.class_label = 2
        self.cls = nn.Linear(self.config.hidden_size, args.class_label)
        # self.test_model = args.test_model
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.gamma_focal > 0:
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)

        
    def forward(self,input_data,inference=False):
        # input_ids = 
        outputs = self.roberta(input_data['input_ids'], input_data['attention_mask'], output_hidden_states = True)
        # pooler = outputs.pooler_output 
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        h12 = hidden_states[-1]
        h11 = hidden_states[-2]
        h10 = hidden_states[-3]
        h09 = hidden_states[-4]
        
        # cat_hidd = torch.cat([h12,h11,h10,h09],dim=-1)
        # att_hidd = self.att_head(cat_hidd)
        
        h12_mean = torch.mean(h12 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h11_mean = torch.mean(h11 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h10_mean = torch.mean(h10 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h09_mean = torch.mean(h09 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        
#         # 对Claim做mean处理，获取Claim的特征
#         sep_index = (input_data['input_ids'] == 2).nonzero()
#         sep_index = sep_index[:,1].view(-1,3) # bs * 3 一个sequence一定有三个<\s>
#         claim_feat = torch.vstack([ torch.mean(h12[i,sep_index[i,1]+1:sep_index[i,2]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
#         # 对Tweet text做mean处理，后去tweet的特征
#         tweet_feat = torch.vstack([ torch.mean(h12[i,1:sep_index[i,0]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
#         # normalized features
#         claim_feat = claim_feat / claim_feat.norm(dim=1, keepdim=True)
#         tweet_feat = tweet_feat / tweet_feat.norm(dim=1, keepdim=True)
#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_claim = logit_scale * claim_feat @ tweet_feat.t()
#         logits_per_tweet = logits_per_claim.t()

#         # calculate loss
#         # logits_matrix = torch.matmul(features_T, features_V.t())
#         labels = torch.arange(0,logits_per_tweet.size(0)).long().cuda()
#         loss_fc = nn.CrossEntropyLoss()
#         loss_T = loss_fc(logits_per_tweet, labels)
#         loss_C = loss_fc(logits_per_claim, labels)
#         loss_clip = (loss_T+loss_C)/2
        
        
        # cat_output = torch.cat([h12_mean,h11_mean,h10_mean,h09_mean,att_hidd],dim = -1)
        # cat_output = torch.cat([h12_mean,att_hidd],dim = -1)
        logits = self.cls(h12_mean)
        probability = nn.functional.softmax(logits)
        if inference:
            return probability
        loss, accuracy, pred_label_id = self.cal_loss(logits, input_data['label'])
        # if False:
        #     loss += loss_clip*0.1
        return loss, accuracy, pred_label_id
        
        
    # @staticmethod
    def cal_loss(self, logits, label):
        # label = label.squeeze(dim=1)
        if self.args.gamma_focal > 0:
            loss = self.focal_loss(logits, label)
        else:
            loss = F.cross_entropy(logits, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(logits, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        if self.args.use_rdrop:
            return loss, logits, accuracy, pred_label_id
        return loss, accuracy, pred_label_id
    
    
class AttentionHead(nn.Module):
    def __init__(self, cat_size, hidden_size=768):
        super().__init__()
        self.W = nn.Linear(cat_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)        
        
    def forward(self, hidden_states):
        att = torch.tanh(self.W(hidden_states))
        score = self.V(att)
        att_w = torch.softmax(score, dim=1)
        context_vec = att_w * hidden_states
        context_vec = torch.sum(context_vec,dim=1)
        
        return context_vec
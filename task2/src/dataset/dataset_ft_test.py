import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
# from transformers import RobertaTokenizer
# from transformers import BertTokenizer
from transformers import AutoTokenizer


import torch

class FinetuneDataset(Dataset):
    def __init__(self, args, data_path, test_model = False):
        self.args = args
        self.data = pd.read_csv(data_path,sep='\t', quoting=csv.QUOTE_NONE)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_model
        self.stance2label = {"AGAINST":0,"FAVOR":1,"NONE":2}
        self.label2stance = {"0":"AGAINST","1":"FAVOR","2":"NONE"}
        self.premise = args.premise
        # self.premise2label = {"AGAINST":0,"FAVOR":1}
        
    def __getitem__(self,index):
        text = self.data['text'].iloc[index]
        claim = self.data['claim'].iloc[index]
        input_data = self.tokenizer(text,claim,max_length=self.bert_seq_length, \
                                    padding='max_length', \
                                    truncation='longest_first')
        input_ids = torch.tensor(input_data['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(input_data['attention_mask'],dtype=torch.long)
        data = dict(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        if self.test_mode:
            return data
        if self.premise:
            premise = self.data['Premise'].iloc[index]
            label_id = torch.tensor(premise,dtype=torch.long)
        else:
            stance = self.data['Stance'].iloc[index]
            label = self.stance2label[stance]
            label_id = torch.tensor(label,dtype=torch.long)
        data['label'] = label_id
        return data
    def __len__(self):
        return self.data.shape[0]
    
    @classmethod
    def create_dataloaders(cls, args):
        test_dataset = cls(args, args.test_path)
        test_sampler = SequentialSampler(test_dataset)
 
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=test_sampler,
                                    drop_last=False,
                                    pin_memory=True)
        print('The test data length: ',len(test_dataloader))
        
        return test_dataloader


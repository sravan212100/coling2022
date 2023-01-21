COLING2022
Common baseline: Validation set mean_f1: 763 step 1554 The f1 score used by the official website is the f1 macro for the average of the three claims Verify set data distribution: face masks：208 school closures：177 stay at home orders：215 Training set data distribution: face masks 1319 school closures 1050 stay at home orders 1187

Hierarchical learning rate mean_f1: 6803
FGM and other confrontation training
Model parameter averaging such as EMA and SWA
SWA: 7866 Online: 7650
python -m src.train
--gpu_ids='2'
--bert_seq_length=128
--learning_rate=1e-4
--bert_learning_rate=5e-5
--savedmodel_path='data/checkpoint/0703_swa_10'
--max_epochs=10
--swa_start=5
--swa_lr=2e-5

EMA: 7900 Online: 7686
python -m src.train
--gpu_ids='2'
--bert_seq_length=128
--learning_rate=1e-4
--bert_learning_rate=5e-5
--savedmodel_path='data/checkpoint/0703_ema_10_128_fgm'
--max_epochs=10
--ema_start=0
--ema_decay=0.99
对抗训练：7944 线上：7686 python -m src.train
--gpu_ids='2'
--bert_seq_length=128
--learning_rate=1e-4
--bert_learning_rate=5e-5
--savedmodel_path='data/checkpoint/0703_ema_11_128_fgm'
--max_epochs=11
--ema_start=0
--ema_decay=0.99
--fgm=1

mix-up
noise tuning
Tweet vs. CLaim: Can't add directly
Pre-training
Do mask mean validation set for the last layer: 7848 step 1900 online: 765
After adding gradient clipping, the score is reduced
Focal loss offline: 7979 gamma: 0.3 Roberta-Large Model In summary, tricks can reach the offline verification set 8118, and the parameters are the same as base Roberta-large adjusts the learning rate and adds gradient clipping to the classification layer Offline 8282
Continuously pre-trained models
The EPL of FGM is 0.5, up to 8449 offline, and 8414 online If the pre-trained rate is too large, the loss will not go down Focal loss is not very useful for large models and will reduce the score

        python -m src.train \
            --gpu_ids='2'\
            --bert_seq_length=128\
            --learning_rate=1e-4\
            --bert_learning_rate=2e-5\
            --savedmodel_path='data/checkpoint/0706_ema_11_fgm_stance_large_pre'\
            --max_epochs=11\
            --ema_start=0\
            --ema_decay=0.99\
            --fgm=1\
            --bert_dir="roberta-large"\
            --pretrain_model_path="data/pretrain_mlm_nsp/model_epoch_8_loss_1.8660_1998.bin"
Using DigitalepidemiologyLab/Covid-Twitter-Bert as the base model plus all tricks without pre-training Offline: 8406
python -m src.train \
    --gpu_ids='2'\
    --bert_seq_length=128\
    --learning_rate=1e-4\
    --bert_learning_rate=2e-5\
    --savedmodel_path='data/checkpoint/0708_ema_11_fgm_stance_twitter'\
    --max_epochs=4\
    --ema_start=0\
    --ema_decay=0.99\
    --fgm=1\
    --bert_dir="digitalepidemiologylab/covid-twitter-bert"\

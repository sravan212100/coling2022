# COLING2022
普通的baseline：验证集mean_f1:763 step 1554
官网用的f1 score是对于三种Claim平均的f1 macro
验证集数据分布：
face masks：208
school closures：177
stay at home orders：215
训练集数据分布：
face masks	1319	
school closures	1050	
stay at home orders	1187
1. 分层学习率 mean_f1:6803
2. fgm等对抗训练
3. EMA、SWA等模型参数平均
    1. SWA : 7866 线上: 7650
    > python -m src.train \
        --gpu_ids='2'\
        --bert_seq_length=128\
        --learning_rate=1e-4\
        --bert_learning_rate=5e-5\
        --savedmodel_path='data/checkpoint/0703_swa_10'\
        --max_epochs=10\
        --swa_start=5\
        --swa_lr=2e-5
    2. EMA : 7900 线上：7686
    > python -m src.train \
        --gpu_ids='2'\
        --bert_seq_length=128\
        --learning_rate=1e-4\
        --bert_learning_rate=5e-5\
        --savedmodel_path='data/checkpoint/0703_ema_10_128_fgm'\
        --max_epochs=10\
        --ema_start=0\
        --ema_decay=0.99\
    > 对抗训练：7944 线上：7686
    > python -m src.train \
        --gpu_ids='2'\
        --bert_seq_length=128\
        --learning_rate=1e-4\
        --bert_learning_rate=5e-5\
        --savedmodel_path='data/checkpoint/0703_ema_11_128_fgm'\
        --max_epochs=11\
        --ema_start=0\
        --ema_decay=0.99\
        --fgm=1
4. mix-up
5. noise tuning
6. Tweet与CLaim之间的对比loss:无法直接加
7. 预训练
8. 对最后一层做 mask mean 验证集：7848 step 1900 线上：765
9. 加了梯度剪裁后 降分
10. focal loss 线下：7979 gamma：0.3
roberta-large 模型 综上tricks可以达到线下验证集8118，参数和base一样
roberta-large 调节学习率并加上对分类层的梯度裁剪 线下8282
11. 持续预训练模型
    > fgm的epl 为0.5 线下可达 8449 线上8414
    > 如果预训练的rate过大，loss则下不去
    > focal loss对于large模型用处不大，会降分
```shell
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
```
    

12. 使用digitalepidemiologylab/covid-twitter-bert作为base 模型 加上全部tricks无预训练 线下：8406
```shell
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
```
> 不加对抗训练 线下直接：8441 fgm对抗训练好像没有太大的作用

13. 冻结word_embeddings 作用不大 分数不变 冻结embedding 降分
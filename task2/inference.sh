#!/bin/bash
echo "Starting inference..."
# python -m src.inference \
#     --gpu_ids='1'\
#     --bert_seq_length=128\
#     --ckpt_file="../data/checkpoint/model_epoch_5_f1_0.7714_1300.bin"\
#     --ema_start=0\
#     --ema_decay=0.99\
#     --result_file="stance_result.tsv"\
#     --bert_dir="../roberta-large"\
#     --pretrain_model_path="../data/pretrain_mlm_nsp_2/model_epoch_6_loss_1.7711_1400.bin"

python -m src.inference \
    --gpu_ids='2'\
    --bert_seq_length=128\
    --ckpt_file="../data/checkpoint/model_epoch_5_f1_0.7714_1300.bin"\
    --ema_start=0\
    --ema_decay=0.99\
    --result_file="premise_result.tsv"\
    --bert_dir="../roberta-large"\
#     --pretrain_model_path="../data/pretrain_mlm_nsp/model_epoch_8_loss_1.8660_1998.bin"\
    --premise
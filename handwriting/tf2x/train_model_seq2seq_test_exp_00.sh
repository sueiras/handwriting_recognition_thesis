#!/bin/bash
source activate tf2

nohup python ./train_model_seq2seq_tf2x.py \
    --config 'config/remote.cfg' \
    --experiment 'IAM/new_seq2seq_exp00' \
    --data_path '/home/jorge/data/tesis/handwriting/databases/IAM/aachen/words/normalized_all' \
    --x_size 64 \
    --y_size 512 > new_seq2seq_exp00.out &

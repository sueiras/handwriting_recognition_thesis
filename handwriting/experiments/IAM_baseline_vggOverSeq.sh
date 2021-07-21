#!/bin/bash
source activate tf1.13

nohup python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/IAM/IAM_baseline_vggOverSeq_run01' \
    --data_path '/home/jorge/data/tesis/handwriting/databases/IAM/aachen/words/normalized_all' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/brown_lob_20000.pkl' \
    --x_shape 192 \
    --y_shape 48 \
    --seq_decoder_len 19 \
    --x_spaces_ini 0 \
    --convolutional_architecture 'vgg_over_seq' \
    --slides_stride 2 \
    --x_slide_size 10 \
    --dense_size_char_model 1024 \
    --rnn_encoder_type 'LSTM' \
    --dim_lstm 256 \
    --num_layers 2 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '1' \
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 50 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > IAM_baseline_vggOverSeq_run01.out &


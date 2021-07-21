#!/bin/bash
source activate tf1.13


#OSBORNE_baseline
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_run02' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/normalized02' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/osborne_trn_val_lexicon_02.pkl' \
    --x_shape 192 \
    --y_shape 48 \
    --seq_decoder_len 19 \
    --x_spaces_ini 0 \
    --convolutional_architecture 'lenet_over_seq' \
    --slides_stride 2 \
    --x_slide_size 10 \
    --dense_size_char_model 1024 \
    --rnn_encoder_type 'LSTM' \
    --dim_lstm 256 \
    --num_layers 2 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 64 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 100 \
    --exponential_decay_rate 0.98 \
    --min_steps 100 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_run02.out


#OSBORNE_2BIGRU256
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_2BIGRU256_run03' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/normalized02' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/osborne_trn_val_lexicon_02.pkl' \
    --x_shape 192 \
    --y_shape 48 \
    --seq_decoder_len 19 \
    --x_spaces_ini 0 \
    --convolutional_architecture 'lenet_over_seq' \
    --slides_stride 2 \
    --x_slide_size 10 \
    --dense_size_char_model 1024 \
    --rnn_encoder_type 'GRU' \
    --dim_lstm 256 \
    --num_layers 2 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 64 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 100 \
    --exponential_decay_rate 0.98 \
    --min_steps 100 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_2BIGRU256_run03.out






python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_256x64_run02' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/normalized02' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/osborne_trn_val_lexicon_02.pkl' \
    --x_shape 256 \
    --y_shape 64 \
    --seq_decoder_len 19 \
    --x_spaces_ini 0 \
    --convolutional_architecture 'lenet_over_seq' \
    --slides_stride 2 \
    --x_slide_size 10 \
    --dense_size_char_model 1024 \
    --rnn_encoder_type 'LSTM' \
    --dim_lstm 256 \
    --num_layers 2 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 64 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 100 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_256x64_run02.out





python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_3BIGRU256_aug_run01' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/normalized02' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/osborne_trn_val_lexicon_02.pkl' \
    --x_shape 192 \
    --y_shape 48 \
    --seq_decoder_len 19 \
    --x_spaces_ini 0 \
    --convolutional_architecture 'lenet_over_seq' \
    --slides_stride 2 \
    --x_slide_size 10 \
    --dense_size_char_model 1024 \
    --rnn_encoder_type 'GRU' \
    --dim_lstm 256 \
    --num_layers 3 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 64 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 100 \
    --exponential_decay_rate 0.98 \
    --min_steps 100 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_3BIGRU256_aug_run01.out
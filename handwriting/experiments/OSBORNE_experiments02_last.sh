#!/bin/bash
source activate tf1.13


#OSBORNE_baseline_1BILSTM256
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_1BILSTM256_run01' \
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
    --num_layers 1 \
    --bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_1BILSTM256_run01.out


#OSBORNE_baseline_1LSTM256
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_1ILSTM256_run01' \
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
    --num_layers 1 \
    --no-bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_1ILSTM256_run01.out


#OSBORNE_baseline_2BIGRU256
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_2BIGRU256_run01' \
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
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_2BIGRU256_run01.out


#OSBORNE_baseline_2GRU256
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_2GRU256_run01' \
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
    --no-bidirectional \
    --num_heads 1 \
    --cuda_device '2' \
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_2GRU256_run01.out




#OSBORNE_baseline_original
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_original_run01' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/sources02' \
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
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_original_run01.out


#OSBORNE_baseline_128x32
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_128x32_run01' \
    --data_path '/home/ubuntu/data/tesis/handwriting/databases/osborne/normalized02' \
    --dictionary_pickle_path '/home/ubuntu/data/handwriting/tesis/osborne_trn_val_lexicon_02.pkl' \
    --x_shape 128 \
    --y_shape 32 \
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
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --teacher_forcing \
    > osborne02_baseline_128x32_run01.out



#OSBORNE_baseline_noTeacher
python ../tf1x/seq2seq_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/osborne/osborne02_baseline_noTeacher_run01' \
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
    --batch_size 128 \
    --dropout_value 0.5 \
    --learning_rate 0.001 \
    --exponential_decay_step 400 \
    --exponential_decay_rate 0.98 \
    --min_steps 40 \
    --max_steps 1000 \
    --early_stopping_steps 20 \
    --lambda_l2_reg 0.0001 \
    --no-data_augmentation \
    --no-teacher_forcing \
    > osborne02_baseline_noTeacher_run01.out


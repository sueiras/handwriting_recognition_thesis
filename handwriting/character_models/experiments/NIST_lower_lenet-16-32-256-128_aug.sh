#!/bin/bash
source activate tf1.13

nohup python ../character_models_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/character/NIST/NIST_lower_lenet-16-32-256-128_aug_run01' \
    --dataset 'NIST' \
    --char_list 'abcdefghijklmnopqrstuvwxyz' \
    --image_size 0 \
    --convolutional_architecture 'lenet' \
    --lenet_conv1_filters 16 \
    --lenet_conv2_filters 32 \
    --vgg_num_blocks 2 \
    --vgg_num_feature_maps_ini 16 \
    --sz_ly0_filters 16 \
    --nb_res_filters 8 \
    --nb_res_stages 1 \
    --dense_size1 256 \
    --dense_size2 128 \
    --cuda_device 2 \
    --max_epochs 400 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --early_stopping_steps 20 \
    --data_augmentation \
    > NIST_lower_lenet-16-32-256-128_aug_run01.out &


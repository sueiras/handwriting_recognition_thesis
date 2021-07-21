#!/bin/bash
source activate tf1.13

nohup python ../character_models_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/character/NIST/NIST_lower_resnet-16-8-2-256-128_aug_run01' \
    --dataset 'NIST' \
    --char_list 'abcdefghijklmnopqrstuvwxyz' \
    --image_size 0 \
    --convolutional_architecture 'resnet' \
    --lenet_conv1_filters 16 \
    --lenet_conv2_filters 32 \
    --vgg_num_blocks 2 \
    --vgg_num_feature_maps_ini 16 \
    --sz_ly0_filters 16 \
    --nb_res_filters 8 \
    --nb_res_stages 2 \
    --dense_size1 256 \
    --dense_size2 128 \
    --cuda_device 1 \
    --max_epochs 400 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --early_stopping_steps 20 \
    --data_augmentation \
    > NIST_lower_resnet-16-8-2-256-128_aug_run01.out &



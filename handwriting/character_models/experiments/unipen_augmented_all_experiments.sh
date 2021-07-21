#!/bin/bash
source activate tf1.13

python ../character_models_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/character/unipen_augmented/unipen_all_vgg-3-64-512-256_aug_run02' \
    --dataset 'unipen_augmented' \
    --char_list '' \
    --image_size 0 \
    --convolutional_architecture 'vgg' \
    --lenet_conv1_filters 20 \
    --lenet_conv2_filters 50 \
    --vgg_num_blocks 3 \
    --vgg_num_feature_maps_ini 64 \
    --sz_ly0_filters 16 \
    --nb_res_filters 8 \
    --nb_res_stages 3 \
    --dense_size1 512 \
    --dense_size2 256 \
    --cuda_device 1 \
    --max_epochs 400 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --early_stopping_steps 20 \
    --data_augmentation \
    > unipen_all_vgg-3-64-512-256_aug_run02.out


python ../character_models_train.py \
    --experiment '/home/ubuntu/data/handwriting/experiments/character/unipen_augmented/unipen_all_resnet-16-8-3-512-256_aug_run02' \
    --dataset 'unipen_augmented' \
    --char_list '' \
    --image_size 0 \
    --convolutional_architecture 'resnet' \
    --lenet_conv1_filters 20 \
    --lenet_conv2_filters 50 \
    --vgg_num_blocks 3 \
    --vgg_num_feature_maps_ini 64 \
    --sz_ly0_filters 16 \
    --nb_res_filters 8 \
    --nb_res_stages 3 \
    --dense_size1 512 \
    --dense_size2 256 \
    --cuda_device 1 \
    --max_epochs 400 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --early_stopping_steps 20 \
    --data_augmentation \
    > unipen_all_resnet-16-8-3-512-256_aug_run02.out




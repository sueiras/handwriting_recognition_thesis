#! /usr/bin/env python

# Version inicial



from __future__ import print_function

import argparse
import codecs
import datetime
import logging
import os
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from tensorflow.keras import callbacks, layers, optimizers, regularizers

from load_data import get_mnist, get_NIST, get_TICH, get_unipen, transform_dataset

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


def encode_target_hex(target):
    ''' Encode text from ascii to hex
    '''
    return codecs.encode(target.encode('utf-8'), "hex").decode('ascii')


def decode_target_hex(target_encoded):
    ''' Decode text from hex to ascii
    '''
    return codecs.decode(target_encoded, "hex").decode('utf-8')





# Parameters
# ==================================================
DATA_PATH = '/home/ubuntu/data/handwriting/databases'


args = argparse.ArgumentParser()

# Paths parameters
args.add_argument("--experiment", type=str, default='/home/ubuntu/data/handwriting/experiments/character/sample', help="Experiment absolute path")
args.add_argument("--dataset", type=str, default='mnist', help="dataset")
args.add_argument("--char_list", type=str, default='', help="char_list")

#Data parameters
args.add_argument("--image_size", type=int, default=0, help="image_size")
args.add_argument("--multiplier_sample", type=int, default=2, help="multiplier_sample for unipen augmented")


# Architecture parameters
args.add_argument("--convolutional_architecture", type=str, default='lenet',
    help="convolutional_architecture: lenet, vgg, resnet"
)
args.add_argument("--dense_size1", type=int, default=1024, help="dense_size1 (default: 1024)")
args.add_argument("--dense_size2", type=int, default=512, help="dense_size2 (default: 512)")

args.add_argument("--sz_ly0_filters", type=int, default=32, help="sz_ly0_filters (default: 32)")
args.add_argument("--nb_res_filters", type=int, default=16, help="nb_res_filters (default: 16)")
args.add_argument("--nb_res_stages", type=int, default=1, help="nb_res_stages (default: 3)")

args.add_argument("--lenet_conv1_filters", type=int, default=6, help="lenet_conv1_filters (default: 8)")
args.add_argument("--lenet_conv2_filters", type=int, default=16, help="lenet_conv2_filters (default: 16)")

args.add_argument("--vgg_num_blocks", type=int, default=2, help="vgg_num_blocks (default: 2)")
args.add_argument("--vgg_num_feature_maps_ini", type=int, default=32, help="vgg_num_feature_maps_ini (default: 32)")



# Training parameters
args.add_argument("--cuda_device", type=str, default='0', help="GPU device (default 0)")
args.add_argument("--batch_size", type=int, default=256, help="Batch Size (default: 256)")
args.add_argument("--max_epochs", type=int, default=300, help="max_epochs (default: 300)")
args.add_argument("--dropout_value", type=float, default=0.5, help="dropout_value (default: 0.5)")
args.add_argument("--learning_rate", type=float, default=0.01, help="learning rate (default: 0.001)")
args.add_argument("--early_stopping_steps", type=int, default=20, help="early_stopping_steps (defaults 20 - max: min_steps parameter)")

args.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', help='Apply data augmentation in train' )
args.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false', help="Don't apply data augmentation in train (default)")
args.set_defaults(data_augmentation=False)


FLAGS, unparsed = args.parse_known_args()



# logging
program_name = 'character_train'
logger = logging.getLogger(program_name)
logger.setLevel(logging.DEBUG)

#logging.basicConfig()
know_time = datetime.datetime.now()
log_name = '_'.join([
    program_name,
    str(know_time.year),
    str(know_time.month).zfill(2),
    str(know_time.day).zfill(2),
    str(know_time.hour).zfill(2),
    str(know_time.minute).zfill(2)+".log"
])

if not os.path.exists(FLAGS.experiment):
    os.makedirs(FLAGS.experiment)
hdlr = logging.FileHandler(os.path.join(FLAGS.experiment, log_name))
logger.addHandler(hdlr)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
hdlr.setFormatter(formatter)

# Check parameters
logger.info(f"Parameters: {FLAGS}")





#Limit GPU cards
#==========================
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

gpu_options = tf.GPUOptions(allow_growth = False)


# Load data
# ==================================================
if FLAGS.dataset == 'mnist':
    logger.info('Reading mnist database. char_list and image_size not available')
    id_trn, X_trn, y_trn, id_val, X_val, y_val, id_tst, X_tst, y_tst, y_labels_dict = get_mnist()

elif FLAGS.dataset == 'TICH':
    logger.info(f'Reading TICH database. Size: {FLAGS.image_size}. char_list not available')
    id_trn, X_trn, y_trn, id_val, X_val, y_val, id_tst, X_tst, y_tst, y_labels_dict = get_TICH(
        DATA_PATH,
        size=FLAGS.image_size
)

elif FLAGS.dataset == 'NIST':
    logger.info(f'Reading NIPS database. Size: {FLAGS.image_size}. char_list: {FLAGS.char_list}')
    id_trn, X_trn, y_trn, id_val, X_val, y_val, id_tst, X_tst, y_tst, y_labels_dict = get_NIST(
        DATA_PATH + '/NIST/by_class/',
        char_list=list(FLAGS.char_list),
        size=FLAGS.image_size
    )

elif FLAGS.dataset == 'unipen':
    logger.info(f'Reading unipen database. Size: {FLAGS.image_size}. char_list: {FLAGS.char_list}')
    id_trn, X_trn, y_trn, id_val, X_val, y_val, id_tst, X_tst, y_tst, y_labels_dict = get_unipen(
        DATA_PATH + '/handwritting_characters_database',
        char_list=list(FLAGS.char_list),
        size=FLAGS.image_size
    )

elif FLAGS.dataset == 'unipen_augmented':
    logger.info(f'Reading unipen database. Size: {FLAGS.image_size}. char_list: {FLAGS.char_list}')

    if FLAGS.image_size not in [0, 64]:
        logger.error(f'Error size must be 0 or 64 to use the unipen_augmented dataset. Size: {FLAGS.image_size}')
        exit(1)

    id_trn, X_trn, y_trn, id_val, X_val, y_val, id_tst, X_tst, y_tst, y_labels_dict = get_unipen(
        DATA_PATH + '/handwritting_characters_database',
        char_list=list(FLAGS.char_list),
        size=FLAGS.image_size
    )
    X_trn, y_trn = transform_dataset(X_trn, y_trn, y_labels_dict, multiplier_sample=FLAGS.multiplier_sample)

else:
    logger.error(f'Error in dataset parameter: {FLAGS.dataset}')
    exit(1)

logger.info(f'Readed data - X_trn.shape: {X_trn.shape}. X_val.shape: {X_val.shape}. X_tst.shape: {X_tst.shape}')
    



# Convolutional layers
# ==================================================
def residual_layer(input_tensor, nb_in_filters=64, nb_bottleneck_filters=16, filter_sz=3, stage=0, reg=0.0):

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage>1: # first activation is just after conv1
        x = layers.BatchNormalization(axis=-1, name='bn'+ str(stage)+'a')(input_tensor)
        x = layers.Activation('relu', name='relu'+str(stage)+'a')(x)
    else:
        x = input_tensor

    x = layers.Conv2D(nb_bottleneck_filters, (1, 1),
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      use_bias=False,
                      name='conv'+str(stage)+'a')(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = layers.BatchNormalization(axis=-1, name='bn'+ str(stage)+'b')(x)
    x = layers.Activation('relu', name='relu'+str(stage)+'b')(x)
    x = layers.Conv2D(nb_bottleneck_filters, (filter_sz, filter_sz),
                      padding='same',
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      use_bias = False,
                      name='conv'+str(stage)+'b')(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = layers.BatchNormalization(axis=-1, name='bn'+ str(stage)+'c')(x)
    x = layers.Activation('relu', name='relu'+str(stage)+'c')(x)
    x = layers.Conv2D(nb_in_filters, (1, 1),
                      kernel_initializer='glorot_normal',
                      kernel_regularizer=regularizers.l2(reg),
                      name='conv'+str(stage)+'c')(x)

    # merge
    x = layers.add([x, input_tensor], name='add'+str(stage))

    return x



def convolution_layers_resnet(
    input_image,
    sz_ly0_filters=32,
    nb_res_filters=16,
    nb_res_stages=3
    ):

    logger.info('Convolutional architecture: Resnet')
    nb_ly0_filters = 3
    nb_ly0_stride = 1
    sz_res_filters = 3
    
    # Complete example: 92% of accuracy
    #sz_ly0_filters, nb_ly0_filters, nb_ly0_stride = (128,3,2)
    #sz_res_filters, nb_res_filters, nb_res_stages = (3,32,25)

    # Initial conv layer
    x = layers.Conv2D(sz_ly0_filters, (nb_ly0_filters,nb_ly0_filters),
                    strides=(nb_ly0_stride, nb_ly0_stride), padding='same', 
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(1.e-4),
                    use_bias=False, name='conv0')(input_image)

    x = layers.BatchNormalization(axis=-1, name='bn0')(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='maxp0')(x)

    # Resnet layers
    for stage in range(1, nb_res_stages+1):
        x = residual_layer(x, nb_in_filters=sz_ly0_filters, nb_bottleneck_filters=nb_res_filters,
                        filter_sz=sz_res_filters, stage=stage, reg=0.0)

    # Complete last resnet layer    
    x = layers.BatchNormalization(axis=-1, name='bnF')(x)
    x = layers.Activation('relu', name='reluF')(x)

    return x



def convolution_layers_lenet(
    input_image,
    lenet_conv1_filters=8,
    lenet_conv2_filters=16
    ):

    logger.info('Convolutional architecture: Lenet')
    
    # conv 1 layer
    x = layers.Conv2D(lenet_conv1_filters, (5,5), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv0'
    )(input_image)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp0')(x)

    # Conv 2 layer
    x = layers.Conv2D(lenet_conv2_filters, (5,5), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv1'
    )(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp1')(x)

    return x





def convolution_layers_vgg(
    input_image,
    vgg_num_blocks=2,
    vgg_num_feature_maps_ini=32
    ):

    logger.info('Convolutional architecture: VGG')
    
    x = input_image
    vgg_num_feature_maps = vgg_num_feature_maps_ini
    for i in range(vgg_num_blocks):
        # VGG 1
        x = layers.Conv2D(vgg_num_feature_maps, (3,3), strides=(1, 1), padding='same', 
            kernel_initializer='glorot_normal',
            kernel_regularizer=regularizers.l2(1.e-4),
            use_bias=False, name='conv0'+str(i)
        )(x)
        x = layers.Activation('relu', name='relu0'+str(i))(x)

        x = layers.Conv2D(vgg_num_feature_maps, (3,3), strides=(1, 1), padding='same', 
            kernel_initializer='glorot_normal',
            kernel_regularizer=regularizers.l2(1.e-4),
            use_bias=False, name='conv1'+str(i)
        )(x)
        x = layers.Activation('relu', name='relu1'+str(i))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp0'+str(i))(x)

        vgg_num_feature_maps = vgg_num_feature_maps * 2

    return x




# Keras model
print('Build model ...')
input_images = layers.Input(shape=(X_trn.shape[1], X_trn.shape[2], 1), name='input')

#input_images_channel = layers.Reshape([X_trn.shape[1], X_trn.shape[2], 1], name='reshape')(input_images) # Add color channel


if FLAGS.convolutional_architecture == 'lenet':
    conv_out = convolution_layers_lenet(
        input_images,
        lenet_conv1_filters=FLAGS.lenet_conv1_filters,
        lenet_conv2_filters=FLAGS.lenet_conv2_filters
        )

elif FLAGS.convolutional_architecture == 'vgg':
    conv_out = convolution_layers_vgg(
        input_images,
        vgg_num_blocks=FLAGS.vgg_num_blocks,
        vgg_num_feature_maps_ini=FLAGS.vgg_num_feature_maps_ini
    )

elif FLAGS.convolutional_architecture == 'resnet':
    conv_out = convolution_layers_resnet(
        input_images,
        sz_ly0_filters=FLAGS.sz_ly0_filters,
        nb_res_filters=FLAGS.nb_res_filters,
        nb_res_stages=FLAGS.nb_res_stages
    )

else:
    logger.error('ERROR in convolutional_architecture parameter', FLAGS.convolutional_architecture)
    exit(11)

# Flatten
conv_out = layers.Flatten(name='Flatten')(conv_out)

#Dense 1
conv_out = layers.Dense(FLAGS.dense_size1, activation='relu', name='dense1')(conv_out)
conv_out = layers.Dropout(FLAGS.dropout_value)(conv_out)

#Dense 2
conv_out = layers.Dense(FLAGS.dense_size2, activation='relu', name='dense2')(conv_out)
conv_out = layers.Dropout(FLAGS.dropout_value)(conv_out)


# Classification layer
output = layers.Dense(len(y_labels_dict), activation='softmax', name='output')(conv_out)







# Create the model
char_model = tf.keras.Model(inputs=input_images, outputs=output)

logger.info(char_model.summary())

RMSprop_optimizer = optimizers.RMSprop(lr=FLAGS.learning_rate)
char_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop_optimizer, metrics=['accuracy'])

EarlyStopping_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=FLAGS.early_stopping_steps,
    restore_best_weights=True
)

TensorBoard_callback = callbacks.TensorBoard(log_dir=FLAGS.experiment)


# Fit the model
if FLAGS.data_augmentation:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 15,       # 15º of random rotation
        width_shift_range = 0.20,  # 20% of random translation width
        height_shift_range = 0.20, # 20% of random translation height
        shear_range = 0.15,        # 5º of shear
        zoom_range = 0.20)         # +- 20% of zoom 

    hist = char_model.fit_generator(
        datagen.flow(X_trn, y_trn, batch_size=FLAGS.batch_size),
        verbose=2,
        steps_per_epoch=int(X_trn.shape[0]/FLAGS.batch_size),
        epochs=FLAGS.max_epochs,
        callbacks=[EarlyStopping_callback, TensorBoard_callback],
        validation_data=(X_val, y_val)
    )
else:
    hist = char_model.fit(
        X_trn,
        y_trn,
        verbose=2,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.max_epochs,
        callbacks=[EarlyStopping_callback, TensorBoard_callback],
        validation_data=(X_val, y_val)
    )


#History
logger.info(f'Train history: {hist.history}')
logger.info(f'Num epochs: {len(hist.history["loss"])}')
pickle.dump( hist.history, open( FLAGS.experiment + "/train_history.pkl", "wb" ) )


#Save model
char_model.save(FLAGS.experiment + '/best_model.h5py')  # creates a HDF5 file 
#model = tf.keras.models.load_model(FLAGS.experiment + '/best_model.h5py')

# Save decode_target
pickle.dump( y_labels_dict, open( FLAGS.experiment + "/target_decoder.pkl", "wb" ) )



# Accuracy measures of the best model over train, valid and test data
logger.info('=============================')
logger.info('       BEST MODEL RESULTS    ')
logger.info('=============================\n')

def evaluate_dataset(model, ids, X, y):

    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)
    y_score = np.sort(y_prob, axis=1)[:,-1]

    logger.info(f"Accuracy score: {accuracy_score(y, y_pred)}")
    logger.info(f"Confusion matrix: {confusion_matrix(y, y_pred)}")

    target_names = list(y_labels_dict.values())
    logger.info(f"classification_report: {classification_report(y, y_pred, labels=range(1,len(target_names)+1), target_names=target_names, zero_division=0)}")
    #logger.info(f"roc_auc_score: {roc_auc_score(y, y_prob, multi_class='ovr')}")

    details_df = pd.DataFrame(
        list(zip(ids, y, y_pred, y_score, y_prob)),
        columns=['id', 'y', 'y_pred', 'y_score', 'y_prob']
    )
    details_df['y'] = details_df['y'].map(y_labels_dict)
    details_df['y_pred'] = details_df['y_pred'].map(y_labels_dict)

    return details_df

logger.info(f"Evaluate train dataset")
logger.info('=============================')
details_trn_df = evaluate_dataset(char_model, id_trn, X_trn, y_trn)
details_trn_df.to_pickle(os.path.join(FLAGS.experiment, 'score_trn_df.pkl'))

logger.info(f"Evaluate validation dataset")
logger.info('=============================')
details_val_df = evaluate_dataset(char_model, id_val, X_val, y_val)
details_val_df.to_pickle(os.path.join(FLAGS.experiment, f'score_val_df.pkl'))

logger.info(f"Evaluate test dataset")
logger.info('=============================')
details_tst_df = evaluate_dataset(char_model, id_tst, X_tst, y_tst)
details_tst_df.to_pickle(os.path.join(FLAGS.experiment, f'score_tst_df.pkl'))

logger.info('Done!')


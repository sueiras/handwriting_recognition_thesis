#! /usr/bin/env python

# Version actual.
# Capas de keras.
# Data augmentation en .py aparte




from __future__ import print_function

import codecs
import datetime
import glob
import logging
import os
import pickle
import random
import sys
import time
import traceback

import cv2
import editdistance
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from tensorflow.contrib.keras import layers, regularizers

from data_augmentation import img_augmented


def encode_target_hex(target):
    ''' Encode text from ascii to hex
    '''
    return codecs.encode(target.encode('utf-8'), "hex").decode('ascii')


def decode_target_hex(target_encoded):
    ''' Decode text from hex to ascii
    '''
    return codecs.decode(target_encoded, "hex").decode('utf-8')



def adjust_image(img_file, x_size=192, y_size=48):
    
    im = Image.open(img_file).convert('L')  # read and convert to grayscale
    img = np.asarray(im, dtype=np.float32)
            
    # ajuste de altura
    y, x = img.shape
    if y_size is not(None):
        img = cv2.resize(img, (max(2,int(x*(y_size/y))), y_size))

    # Recorte derecha e izquierda
    true_points = np.argwhere(img)
    if len(true_points)>0:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
         # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        if bottom_right[1] - top_left[1] > 2:
            img = img[:, top_left[1]:bottom_right[1]+1]

    # Ajuste de anchura
    y, x = img.shape
    if x < x_size:
        img = np.concatenate([img, np.zeros([y, x_size - x])], axis=1)
    else:
        img = cv2.resize(img, (x_size, y_size))
        
    return img / 255.




def read_word_image(path, f, encoder_dict, x_size=192, y_size=48, target_max_size=19, x_spaces_ini=0):
    ''' Read image from file and get the target from the image name

    '''
    # adjust and resize to the final size
    img_adjusted = adjust_image(os.path.join(path, f), x_size=x_size, y_size=y_size)

    if img_adjusted != []:
        try:
            #Calculate image_len
            image_len = np.max(np.nonzero(np.max(img_adjusted, axis=0)))
            # Target
            file_name = '.'.join(f.split('.')[:-1])
            hex_name = file_name.split('_')[-1] # extract hexadecimal part
            target_c = decode_target_hex(hex_name) # convert to string
            target_ini = [encoder_dict[k] for k in target_c] # encode to
            if len(target_ini)>target_max_size: # Pendiente de resolver mejor
                target_ini = target_ini[:target_max_size]
            target_len = len(target_ini)
            target = np.ones([target_max_size], dtype=np.uint8)*(-1)
            target[:target_len] = target_ini
            return img_adjusted, list(target), image_len, target_len
        except:
            print(f)
            return [], None, None, None
    else:
        return [], None, None, None






def load_data_in_memory(images_path, encoder_dict, x_size=192, y_size=48, x_spaces_ini=0, sample_size=0):
    '''Load data in memory
    '''
    pathlist = glob.glob(os.path.join(images_path, '*.png'))

    # Muestreo para sintéticas
    if sample_size>0:
        pathlist = np.random.choice(pathlist, size=sample_size, replace=False)


    filenames_list = []
    img_list = []
    img_l_list = []
    target_list = []
    target_l_list = []
    for f in pathlist:
        try:
            img, target, img_l, target_l = read_word_image(
                images_path,
                os.path.basename(f),
                encoder_dict,
                x_size=x_size,
                y_size=y_size,
                x_spaces_ini=x_spaces_ini)
            if len(target)>0:
                filenames_list += [os.path.basename(f)]
                img_list += [img]
                img_l_list += [img_l]
                target_list += [target]
                target_l_list += [target_l]
            else:
                logger.info('Error reading the file %s. len(target)=0', f)
        except:
            logger.info('Error reading the file %s', f)

    return np.array(img_list, dtype=np.float32),  np.array(img_l_list, dtype=np.uint8), \
           np.array(target_list, dtype=np.uint8), np.array(target_l_list, dtype=np.uint8), np.array(filenames_list)





# data_utils.py
#=============================


def create_decoder_dir():
    char_list = [
        32, #   # 20
        33, # ! # 21
        34, # " # 22
        35, # # # 23
        36, # $ # 24
        37, # % # 25
        38, # & # 26
        39, # ' # 27
        40, # ( # 28
        41, # ) # 29
        42, # * # 2a
        43, # + # 2b
        44, # , # 2c
        45, # - # 2d
        46, # . # 2e
        47, # / # 2f
        48, # 0 # 30
        49, # 1 # 31
        50, # 2 # 32
        51, # 3 # 33
        52, # 4 # 34
        53, # 5 # 35
        54, # 6 # 36
        55, # 7 # 37
        56, # 8 # 38
        57, # 9 # 39
        58, # : # 3a
        59, # ; # 3b
        60, # < # 3c
        61, # = # 3d
        62, # > # 3e
        63, # ? # 3f
        64, # @ # 40
        65, # A # 41
        66, # B # 42
        67, # C # 43
        68, # D # 44
        69, # E # 45
        70, # F # 46
        71, # G # 47
        72, # H # 48
        73, # I # 49
        74, # J # 4a
        75, # K # 4b
        76, # L # 4c
        77, # M # 4d
        78, # N # 4e
        79, # O # 4f
        80, # P # 50
        81, # Q # 51
        82, # R # 52
        83, # S # 53
        84, # T # 54
        85, # U # 55
        86, # V # 56
        87, # W # 57
        88, # X # 58
        89, # Y # 59
        90, # Z # 5a
        91, # [ # 5b
        92, # \ # 5c
        93, # ] # 5d
        95, # _ # 5f
        97, # a # 61
        98, # b # 62
        99, # c # 63
        100, # d # 64
        101, # e # 65
        102, # f # 66
        103, # g # 67
        104, # h # 68
        105, # i # 69
        106, # j # 6a
        107, # k # 6b
        108, # l # 6c
        109, # m # 6d
        110, # n # 6e
        111, # o # 6f
        112, # p # 70
        113, # q # 71
        114, # r # 72
        115, # s # 73
        116, # t # 74
        117, # u # 75
        118, # v # 76
        119, # w # 77
        120, # x # 78
        121, # y # 79
        122, # z # 7a
        123, # { # 7b
        125, # } # 7d
        161, # ¡ # c2a1
        191, # ¿ # c2bf
        193, # Á # c381
        199, # Ç # c387
        201, # É # c389
        205, # Í # c38d
        209, # Ñ # c391
        211, # Ó # c393
        218, # Ú # c39a
        224, # à # c3a0
        225, # á # c3a1
        226, # â # c3a2
        231, # ç # c3a7
        232, # è # c3a8
        233, # é # c3a9
        234, # ê # c3aa
        237, # í # c3ad
        241, # ñ # c3b1
        243, # ó # c3b3
        250, # ú # c3ba
        252, # ü # c3bc
        8364 # €
    ]
    char_list =sorted(list(set(char_list)))

    decoder_dict = {}
    for i, c in enumerate(char_list):
        decoder_dict[i] = chr(c)

    # Complete decoding dictionary
    num_characters = len(decoder_dict)
    PAD_ID = num_characters
    GO_ID = num_characters+1
    EOL_ID = num_characters+2
    decoder_dict[PAD_ID]='-PAD'
    decoder_dict[GO_ID]='GO'
    decoder_dict[EOL_ID]='-EOL'
    
    return PAD_ID, GO_ID, EOL_ID, decoder_dict




def decoder_dict_database(iam_patches):
    '''
    Load decoder dictionary for IAM database
    '''
    #Recover decoding dictionary
    with h5py.File(iam_patches, "r") as hdf5_f:
        keys_dict = np.copy(hdf5_f["target_dict_keys"])
        values_dict = np.copy(hdf5_f["target_dict_values"])
    decoder_dict={}
    for i, key in enumerate(keys_dict):
        decoder_dict[key] = values_dict[i].decode('UTF-8')

    # Complete decoding dictionary
    num_characters = len(decoder_dict)
    PAD_ID = num_characters
    GO_ID = num_characters+1
    EOL_ID = num_characters+2
    decoder_dict[PAD_ID]='-PAD'
    decoder_dict[GO_ID]='GO'
    decoder_dict[EOL_ID]='-EOL'

    return PAD_ID, GO_ID, EOL_ID, decoder_dict



def decode_text(text_array, decoder_dict):
    '''
    Decode the target from numbers to words
    '''
    text = ''
    eol_code = len(decoder_dict)-1
    ind_eol = False
    for c in text_array:
        if ind_eol==False:
            text += decoder_dict[c]
        if c==eol_code:
            ind_eol=True
    return text
# logger.info(decode_text(np.array([44, 53, 67, 57, 71, 53, 81, 81], dtype=np.uint8))) #Sandra





#Generate slides of the image
def generate_images(img_batch, img_len_batch, x_slide_size = 28, slides_stride = 2, augmentation=False):

    images_batch = []
    slides_len_batch = []
    # Convert img_batch in a sequence of frames and calculate slides_len_batch
    for n_img, img in enumerate(img_batch):
        #Data augmentation
        if augmentation:
            try:
                img_aug = img_augmented(img)
                if img_aug.max()>0:
                    images_batch += [img_aug]
                else:
                    logger.info("Augmentation return empty image")
                    images_batch += [img]
            except Exception as e:
                logger.info(f'Error in data augmentation. Exception {e}')
                traceback.print_exc()
                images_batch += [img]
        else:
            images_batch += [img]
        #Calculate slides_len_batch as the number of slides to get
        max_slides = int((img.shape[1] - x_slide_size)/float(slides_stride))
        num_slides = max(2,min(max_slides, 1 + int((img_len_batch[n_img] - x_slide_size)/float(slides_stride))))
        slides_len_batch += [num_slides]

    return np.array(images_batch), np.array(slides_len_batch)



def generate_target(y_ini, y_len, seq_length=19, num_classes=81+3):

    #Create vars: target, dec_inp and weigth
    batch_size = y_ini.shape[0]
    decoder_inputs = np.zeros([batch_size, seq_length+1, num_classes], dtype=np.float32)
    weights = np.zeros([batch_size, seq_length+1], dtype=np.float32)
    targets = np.zeros([batch_size, seq_length+1], dtype=np.uint16)
    for batch_i in range(batch_size):
        for char_pos in range(seq_length+1):
            if char_pos == 0:
                decoder_inputs[batch_i, char_pos, GO_ID] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = int(y_ini[batch_i, char_pos])
            elif char_pos < y_len[batch_i]:
                decoder_inputs[batch_i, char_pos, int(y_ini[batch_i, char_pos-1])] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = int(y_ini[batch_i, char_pos])
            elif char_pos == y_len[batch_i]:
                decoder_inputs[batch_i, char_pos, int(y_ini[batch_i, char_pos-1])] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = EOL_ID
            else:
                decoder_inputs[batch_i, char_pos, PAD_ID] = 1
                weights[batch_i, char_pos] = 0
                targets[batch_i, char_pos] = PAD_ID

    return decoder_inputs, targets, weights




def batch_generator_epoch(
    img,
    img_l,
    target,
    target_l,
    filenames,
    batch_size=256,
    seq_length=19,
    slides_stride=5,
    x_slide_size=28,
    num_classes=81+3,
    augmentation=False
):
    '''
    Generator to one epoch of data
    '''
    #Randomize batches
    data_size = img.shape[0]
    p_index = np.random.permutation(range(0, data_size))
    batch_list=[]
    for b in range(0, data_size, batch_size):
        batch_list += [list(p_index[b:b+batch_size])]

    # Iterate over each batch
    for batch in batch_list:
        #Extract batch data
        filenanes_batch = list(filenames[batch])
        img_b = img[batch]
        img_l_b = img_l[batch]
        target_b = target[batch]
        target_l_b = target_l[batch]

        images_batch, slides_len_batch = generate_images(
            img_b,
            img_l_b,
            x_slide_size = x_slide_size,
            slides_stride = slides_stride,
            augmentation=augmentation
        )

        decoder_inputs_batch, targets_batch, weights_batch = generate_target(
            target_b,
            target_l_b,
            seq_length=seq_length,
            num_classes=num_classes
        )

        yield  images_batch, slides_len_batch, decoder_inputs_batch, targets_batch, weights_batch, img_b, target, filenanes_batch




# Evaluation functions
# ==================================================

# Real test dictionary
def get_real_test_dictionary_set(dataset_patches):
    with h5py.File(dataset_patches, "r") as hdf5_f:
        target_tst = np.copy(hdf5_f["target_tst"])
        target_length_tst = np.copy(hdf5_f["target_length_tst"])
    target_tst_set = set([decode_text(target_tst[i][:target_length_tst[i]], decoder_dict) for i in range(len(target_tst))])
    return target_tst_set



def calculate_wer_cer(real, predict):
    WER = 0
    CER = 0
    n_chars = 0
    errors = 0
    for i in range(len(real)):
        CER += editdistance.eval(real[i], predict[i])
        n_chars += len(real[i])
        if real[i] != predict[i]:
            WER += 1
            errors += 1

    return float(WER)/len(real), float(CER)/n_chars


def evaluate_corpus(real, predict, set_lexicon):
    #for each word in the predict list, find the closed word in the test_set.
    predict_lexicon=[]
    for w in predict:
        min_distance=1000
        closed_word=''
        for wt in set_lexicon:
            if editdistance.eval(w, wt) < min_distance:
                min_distance = editdistance.eval(w, wt)
                closed_word = wt
        predict_lexicon += [closed_word]

    wer, cer = calculate_wer_cer(real, predict_lexicon)

    return wer, cer, predict_lexicon









# Parameters
# ==================================================
import argparse

args = argparse.ArgumentParser()

# Paths parameters
#args.add_argument("--decoder_dict_data", type=str, default='/home/ubuntu/data/handwriting/tesis/rimes_words_48_192.hdf5', help="decoder_dict_data")
args.add_argument("--experiment", type=str, default='/home/ubuntu/data/handwriting/experiments/rimes/sample', help="Experiment absolute path")
args.add_argument("--data_path", type=str, default='/opt/data/tesis/handwriting/databases/rimes/sample', help="data_path")
args.add_argument("--dictionary_pickle_path", type=str, default='/home/ubuntu/data/tesis/rimes_trn_lexicon.pkl', help="dictionary_pickle_path")

#Data parameters
args.add_argument("--x_shape", type=int, default=192, help="x_shape (default: 192)")
args.add_argument("--y_shape", type=int, default=48, help="y_shape (default: 48)")
args.add_argument("--seq_decoder_len", type=int, default=19, help="max_length of a word (default: 19)")
args.add_argument("--x_spaces_ini", type=int, default=0, help="x_spaces_ini (default: 0)")


# Architecture parameters
args.add_argument("--convolutional_architecture", type=str, default='lenet_over_seq',
    help="convolutional_architecture: lenet_over_seq, vgg_over_seq, lenet, vgg or resnet (default lenet_over_seq)"
)
args.add_argument("--x_slide_size", type=int, default=10, help="x_slide_size only _over_seq architectures (default: 10)")
args.add_argument("--slides_stride", type=int, default=2, help="slides_stride only _over_seq architectures (default: 2)")
args.add_argument("--dense_size_char_model", type=int, default=1024, help="dense size of the char model only _over_seq architectures (default: 1024)")


args.add_argument("--rnn_encoder_type", type=str, default='LSTM', help="rnn_encoder_type: LSTM or GRU (default LSTM)")
args.add_argument("--dim_lstm", type=int, default=256, help="dim_lstm (default: 256)")
args.add_argument("--num_layers", type=int, default=2, help="num_layers 1, 2 or 3 (default: 2)")
args.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Use a bidirectional model (default)')
args.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', help='No use a bidirectional model.')
args.set_defaults(bidirectional=True)
args.add_argument("--num_heads", type=int, default=1, help="num_heads (default: 1)")


# Training parameters
args.add_argument("--cuda_device", type=str, default='0', help="GPU device (default 0)")
args.add_argument("--batch_size", type=int, default=256, help="Batch Size (default: 256)")
args.add_argument("--dropout_value", type=float, default=0.5, help="dropout_value (default: 0.5)")
args.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (default: 0.001)")
args.add_argument("--exponential_decay_step", type=int, default=400, help="exponential_decay_step (defaults 400)")
args.add_argument("--exponential_decay_rate", type=float, default=0.98, help="exponential_decay_rate (default 0.98)")
args.add_argument("--min_steps", type=int, default=20, help="min_steps (defaults 20 - min early_stopping_steps)")
args.add_argument("--max_steps", type=int, default=1000, help="max_steps (defaults 1000 - min: min_steps parameter)")
args.add_argument("--early_stopping_steps", type=int, default=20, help="early_stopping_steps (defaults 20 - max: min_steps parameter)")
args.add_argument("--lambda_l2_reg", type=float, default=0.0001, help="lambda_l2_reg (default: 0.0001)")

args.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', help='Apply data augmentation in train' )
args.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false', help="Don't apply data augmentation in train (default)")
args.set_defaults(data_augmentation=False)

args.add_argument('--teacher_forcing', dest='teacher_forcing', action='store_true', help='Apply teacher_forcing in train (default)')
args.add_argument('--no-teacher_forcing', dest='teacher_forcing', action='store_false', help="Don't apply teacher_forcing in train")
args.set_defaults(teacher_forcing=True)




FLAGS, unparsed = args.parse_known_args()



# logging
program_name = 'seq2seq_train'
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




# Data generator parameters
#==========================

#weights_dictionary
weights_dict = {i:1 for i in range(100)}
weights_dict[0] = 10
weights_dict[1] = 5


# Encoder and decoder dict

# Decoder generico
PAD_ID, GO_ID, EOL_ID, decoder_dict = create_decoder_dir()

# Decoder especifico dataset
#PAD_ID, GO_ID, EOL_ID, decoder_dict = decoder_dict_database(FLAGS.decoder_dict_data)

encoder_dict = {}
for e in decoder_dict:
    encoder_dict[decoder_dict[e]]=e


logger.info(f'decoder_dict: {decoder_dict}')
# The last 3 characters of the dict are the special characters PAD, GO and END.
num_classes = len(decoder_dict)
#num_characters = num_classes - 3



trn_dir = os.path.join(FLAGS.data_path, 'trn')
img_trn, img_l_trn, target_trn, target_l_trn, filenames_list_trn = load_data_in_memory(
    trn_dir,
    encoder_dict,
    x_size=FLAGS.x_shape,
    y_size=FLAGS.y_shape,
    x_spaces_ini=FLAGS.x_spaces_ini
)
logger.info(f'Loaded train files: {len(img_trn)}')

val_dir = os.path.join(FLAGS.data_path, 'val')
img_val, img_l_val, target_val, target_l_val, filenames_list_val = load_data_in_memory(
    val_dir,
    encoder_dict,
    x_size=FLAGS.x_shape,
    y_size=FLAGS.y_shape,
    x_spaces_ini=FLAGS.x_spaces_ini
)
logger.info(f'Loaded valid files: {len(img_val)}')

tst_dir = os.path.join(FLAGS.data_path, 'tst')
img_tst, img_l_tst, target_tst, target_l_tst, filenames_list_tst = load_data_in_memory(
    tst_dir,
    encoder_dict,
    x_size=FLAGS.x_shape,
    y_size=FLAGS.y_shape,
    x_spaces_ini=FLAGS.x_spaces_ini
)
logger.info(f'Loaded test files: {len(img_tst)}')



#Limit GPU cards
#==========================
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

gpu_options = tf.GPUOptions(allow_growth = False)



def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'   + name, mean)
        tf.summary.scalar('sttdev/' + name, tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('max/'    + name, tf.reduce_max(var))
        tf.summary.scalar('min/'    + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)




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



def convolution_layers_resnet(input_image):

    logger.info('Convolutional architecture: Resnet')
    
    # Extract patches of original image
    img_input = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel

    sz_ly0_filters, nb_ly0_filters, nb_ly0_stride = (32,3,1)
    sz_res_filters, nb_res_filters, nb_res_stages = (3,16,3)

    # Complete example: 92% of accuracy
    #sz_ly0_filters, nb_ly0_filters, nb_ly0_stride = (128,3,2)
    #sz_res_filters, nb_res_filters, nb_res_stages = (3,32,25)

    # Initial conv layer
    x = layers.Conv2D(sz_ly0_filters, (nb_ly0_filters,nb_ly0_filters),
                    strides=(nb_ly0_stride, nb_ly0_stride), padding='same', 
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(1.e-4),
                    use_bias=False, name='conv0')(img_input)

    x = layers.BatchNormalization(axis=-1, name='bn0')(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='maxp0')(x)
    variable_summaries(x, 'resnet_layer_ini')

    # Resnet layers
    for stage in range(1, nb_res_stages+1):
        x = residual_layer(x, nb_in_filters=sz_ly0_filters, nb_bottleneck_filters=nb_res_filters,
                        filter_sz=sz_res_filters, stage=stage, reg=0.0)
        variable_summaries(x, 'resnet_layer_'+str(stage))

    # Complete last resnet layer    
    x = layers.BatchNormalization(axis=-1, name='bnF')(x)
    x = layers.Activation('relu', name='reluF')(x)


    # Final layer
    x = layers.Permute((2, 1, 3))(x)
    s = x.get_shape()
    x = layers.Reshape((s[1], s[2]*s[3]))(x)

    return x



def convolution_layers_lenet(input_image):

    logger.info('Convolutional architecture: Lenet')
    
    # Extract patches of original image
    img_input = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel

    # conv 1 layer
    x = layers.Conv2D(20, (5,5), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv0'
    )(img_input)

    x = layers.Activation('relu', name='relu0')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp0')(x)
    variable_summaries(x, 'conv0')

    # Conv 2 layer
    x = layers.Conv2D(50, (5,5), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv1'
    )(x)

    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp1')(x)
    variable_summaries(x, 'conv1')

    #La diferencia con los slices es que no hay capa densa

    # Final layer: concat channels by column
    x = layers.Permute((2, 1, 3))(x)
    s = x.get_shape()
    x = layers.Reshape((s[1], s[2]*s[3]))(x)

    return x





def convolution_layers_vgg(input_image):

    logger.info('Convolutional architecture: VGG')
    
    # Extract patches of original image
    img_input = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel

    # VGG 1
    x = layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv0'
    )(img_input)
    x = layers.Activation('relu', name='relu0')(x)

    x = layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv1'
    )(x)
    x = layers.Activation('relu', name='relu1')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp0')(x)
    variable_summaries(x, 'maxp0')

    # VGG 2
    x = layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv2'
    )(x)
    x = layers.Activation('relu', name='relu2')(x)

    x = layers.Conv2D(64, (3,3), strides=(1, 1), padding='same', 
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(1.e-4),
        use_bias=False, name='conv3'
    )(x)
    x = layers.Activation('relu', name='relu3')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name='maxp1')(x)
    variable_summaries(x, 'maxp1')

    #La diferencia con los slices es que no hay capa densa

    # Final layer: concat channels by column
    x = layers.Permute((2, 1, 3))(x)
    s = x.get_shape()
    x = layers.Reshape((s[1], s[2]*s[3]))(x)

    return x



def lenet_over_seq(img_seq, dropout_keep_prob):
    logger.info('-- Lenet  model.')
    #First convolution
    W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1))
    b_conv_1 = tf.Variable(tf.constant(0.1, shape=[20]), name='bias_c1')
    conv1_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1) for x_in in img_seq]
    h_pool1 = [tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv1 in conv1_out]
    variable_summaries(W_conv_1, 'W_conv_1')

    #Second convolution
    W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 20, 50], stddev=0.1))
    b_conv_2 = tf.Variable(tf.constant(0.1, shape=[50]), name='bias_c2')
    conv2_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2) for x_in in h_pool1]
    h_pool2 = [tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv2 in conv2_out]
    variable_summaries(W_conv_2, 'W_conv_2')

    #First dense layer
    h_pool2_flat = [tf.contrib.layers.flatten(hp) for hp in h_pool2]
    dim_pool = h_pool2_flat[0].get_shape()

    W_dense_1 = tf.Variable(tf.truncated_normal([dim_pool[1].value, FLAGS.dense_size_char_model], stddev=0.1))
    b_dense_1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.dense_size_char_model]), name='bias_d1')
    dense_output_1 = [tf.nn.relu(tf.matmul(x_in, W_dense_1) + b_dense_1) for x_in in h_pool2_flat]

    #Dropout over
    #h_fc1_drop = [tf.nn.dropout(h_fc1, dropout_keep_prob) for h_fc1 in dense_output_1]

    return dense_output_1



def VGG_over_seq(img_seq, dropout_keep_prob):
    logger.info('-- VGG  model.')
    #First convolution
    W_conv_11 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
    b_conv_11 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias_c11')
    conv11_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_11, strides=[1, 1, 1, 1], padding='SAME') + b_conv_11) for x_in in img_seq]

    W_conv_12 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    b_conv_12 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias_c12')
    conv12_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_12, strides=[1, 1, 1, 1], padding='SAME') + b_conv_12) for x_in in conv11_out]

    h_pool1 = [tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv1 in conv12_out]
    variable_summaries(W_conv_11, 'W_conv_12')

    #Second convolution
    W_conv_21 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    b_conv_21 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_c21')
    conv21_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_21, strides=[1, 1, 1, 1], padding='SAME') + b_conv_21) for x_in in h_pool1]

    W_conv_22 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    b_conv_22 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_c22')
    conv22_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_22, strides=[1, 1, 1, 1], padding='SAME') + b_conv_22) for x_in in conv21_out]

    h_pool2 = [tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv2 in conv22_out]
    variable_summaries(W_conv_21, 'W_conv_22')

    #First dense layer
    h_pool2_flat = [tf.contrib.layers.flatten(hp) for hp in h_pool2]
    dim_pool = h_pool2_flat[0].get_shape()

    W_dense_1 = tf.Variable(tf.truncated_normal([dim_pool[1].value, FLAGS.dense_size_char_model], stddev=0.1))
    b_dense_1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.dense_size_char_model]), name='bias_d1')
    dense_output_1 = [tf.nn.relu(tf.matmul(x_in, W_dense_1) + b_dense_1) for x_in in h_pool2_flat]

    #Dropout over
    #h_fc1_drop = [tf.nn.dropout(h_fc1, dropout_keep_prob) for h_fc1 in dense_output_1]

    return dense_output_1


def convolution_layers_vgg_over_seq(input_image):

    logger.info('Convolutional architecture: VGG over seq')
    
    image_reshape = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel
    patches  = tf.extract_image_patches(image_reshape, [1, FLAGS.y_shape, FLAGS.x_slide_size, 1],
                                        [1, 1, FLAGS.slides_stride, 1], [1, 1, 1, 1], 'VALID' ) #Dim [b, n_patches, y_patch*x_patch]

    seq_encoder_len = patches.get_shape()[2].value
    patches_reshaped = tf.reshape(patches, [-1, seq_encoder_len, FLAGS.y_shape, FLAGS.x_slide_size]) #Dim [b, n_patches, y_patch, x_patch]

    # Generate input convolutions
    input_convolution = [tf.transpose(t, perm=[0, 2, 3, 1]) for t in tf.split(patches_reshaped, seq_encoder_len, axis=1)]
    #logger.info('-- Input_convolution: ', input_convolution) # List of [b, y_patch, x_patch] of size num_patches


    input_encoder_list = VGG_over_seq(input_convolution, keep_prob)
    input_encoder = tf.stack(input_encoder_list, axis=1)

    return input_encoder


def convolution_layers_lenet_over_seq(input_image):

    logger.info('Convolutional architecture: Lenet over seq')

    image_reshape = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel
    patches  = tf.extract_image_patches(image_reshape, [1, FLAGS.y_shape, FLAGS.x_slide_size, 1],
                                        [1, 1, FLAGS.slides_stride, 1], [1, 1, 1, 1], 'VALID' ) #Dim [b, n_patches, y_patch*x_patch]

    seq_encoder_len = patches.get_shape()[2].value
    patches_reshaped = tf.reshape(patches, [-1, seq_encoder_len, FLAGS.y_shape, FLAGS.x_slide_size]) #Dim [b, n_patches, y_patch, x_patch]

    # Generate input convolutions
    input_convolution = [tf.transpose(t, perm=[0, 2, 3, 1]) for t in tf.split(patches_reshaped, seq_encoder_len, axis=1)]
    #logger.info('-- Input_convolution: ', input_convolution) # List of [b, y_patch, x_patch] of size num_patches

    input_encoder_list = lenet_over_seq(input_convolution, keep_prob)
    input_encoder = tf.stack(input_encoder_list, axis=1)
    
    return input_encoder



def get_recurrent_layer(rnn_type, return_state=True):
    if rnn_type == 'LSTM':
        rnn_encoder_layer = layers.LSTM(
            FLAGS.dim_lstm,
            return_sequences=True,
            return_state=return_state,
            dropout=FLAGS.dropout_value,
            recurrent_initializer='glorot_uniform'
        )
    elif rnn_type == 'GRU':
        rnn_encoder_layer = layers.GRU(
            FLAGS.dim_lstm,
            return_sequences=True,
            return_state=return_state,
            dropout=FLAGS.dropout_value,
            recurrent_initializer='glorot_uniform'
        )
    else: # error
        logger.info(f'ERROR: No valid rnn_encoder_type parameter: {rnn_type}')
        sys.exit(1)
    return rnn_encoder_layer



### Create model
### ==============================================================
dropout_value = FLAGS.dropout_value

graph = tf.Graph()
with graph.as_default():

    #Placeholders
    with tf.name_scope('inputs') as scope:
        input_image = tf.placeholder(np.float32,[None, FLAGS.y_shape, FLAGS.x_shape], name='input_image')

        input_slides_len = tf.placeholder(tf.int32, shape=(None), name='input_word_len')

        input_word_chars = tf.placeholder(tf.float32, shape=(None, FLAGS.seq_decoder_len+1, num_classes),
                                          name="input_word_chars")
        input_decoder = [tf.reshape(t, [-1, num_classes]) for t in tf.split(input_word_chars, FLAGS.seq_decoder_len+1, axis=1)]

        input_targets = tf.placeholder(tf.int32  , shape=[None, FLAGS.seq_decoder_len+1], name='input_targets')
        input_weights = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_decoder_len+1], name='input_weights')

        weights = [tf.reshape(t, [-1]) for t in tf.split(input_weights, FLAGS.seq_decoder_len+1, axis=1 )]
        targets = [tf.reshape(t, [-1]) for t in tf.split(input_targets, FLAGS.seq_decoder_len+1, axis=1 )]

        keep_prob = tf.placeholder(tf.float32)

        is_training = tf.placeholder(tf.bool)

        # If teacher_forcing = True -->  TRAIN: use the real previous output to predict the next output
        # If teacher_forcing = False --> TEST: use the previous predicted output for the next output
        teacher_forcing = tf.placeholder(tf.bool)



    #Transform images to input to the LSTM encoder
    with tf.name_scope('convolutions') as scope:
        logger.info(f'Convoilutional architecture: {FLAGS.convolutional_architecture}')

        if FLAGS.convolutional_architecture == 'resnet':
            input_encoder = convolution_layers_resnet(input_image)

        elif FLAGS.convolutional_architecture == 'vgg':
            input_encoder = convolution_layers_vgg(input_image)

        elif FLAGS.convolutional_architecture == 'lenet':
            input_encoder = convolution_layers_lenet(input_image)

        elif FLAGS.convolutional_architecture == 'vgg_over_seq':
            input_encoder = convolution_layers_vgg_over_seq(input_image)

        elif FLAGS.convolutional_architecture == 'lenet_over_seq':
            input_encoder = convolution_layers_lenet_over_seq(input_image)

        else: 
            logger.info(f'ERROR: No valid convolutional architecture parameter {FLAGS.convolutional_architecture}')
            sys.exit(1)

        variable_summaries(input_encoder, 'input_encoder')


    ## ENCODER
    with tf.name_scope('encoder') as scope:

        if FLAGS.bidirectional: # Bidirectional model

            if FLAGS.num_layers == 1:
                output_bidirectional = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                )(input_encoder, training=is_training)

            elif FLAGS.num_layers == 2:
                x = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                )(input_encoder, training=is_training)
                output_bidirectional = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                )(x, training=is_training)
                
            elif FLAGS.num_layers == 3:
                x = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                )(input_encoder, training=is_training)
                x = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                )(x, training=is_training)
                output_bidirectional = layers.Bidirectional(
                    get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                )(x, training=is_training)

            else: # error
                logger.info(f'ERROR: No valid num_layers parameter {FLAGS.num_layers}')
                sys.exit(1)


            if FLAGS.rnn_encoder_type == 'GRU':
                output_encoder, final_state_encoder_f, _ = output_bidirectional
            else: # if lstm return 4 states [memory state f, carry state f, memory state b, carry state b]
                output_encoder, final_state_encoder_f_m, final_state_encoder_f_c,  _, _ = output_bidirectional
                final_state_encoder_f = [final_state_encoder_f_m, final_state_encoder_f_c]

            # Secuencia de salida de la ultima capa del encoder
            attention_states = output_encoder
            # Se usa el estado final del paso forward cmo estado inicial del decoder
            #initial_state_decoder = final_state_encoder_f
            initial_state_decoder = final_state_encoder_f


        else: # No bidirectional model

            if FLAGS.num_layers == 1:
                rnn_encoder_layer_01 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                output_rnn_encoder = rnn_encoder_layer_01(input_encoder, training=is_training)

            elif FLAGS.num_layers == 2:
                rnn_encoder_layer_01 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                x = rnn_encoder_layer_01(input_encoder, training=is_training)
                rnn_encoder_layer_02 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                output_rnn_encoder = rnn_encoder_layer_02(x, training=is_training)

            elif FLAGS.num_layers == 3:
                rnn_encoder_layer_01 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                x = rnn_encoder_layer_01(input_encoder, training=is_training)
                rnn_encoder_layer_02 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=False)
                x = rnn_encoder_layer_02(x, training=is_training)
                rnn_encoder_layer_03 = get_recurrent_layer(FLAGS.rnn_encoder_type, return_state=True)
                output_rnn_encoder = rnn_encoder_layer_03(x, training=is_training)

            else: # error
                logger.info(f'ERROR: No valid num_layers parameter {FLAGS.num_layers}')
                sys.exit(1)

            if FLAGS.rnn_encoder_type == 'GRU':
                output_encoder, final_state_encoder = output_rnn_encoder
            else: # if lstm return 2 states [memory state, carry state]
                output_encoder, final_state_m, final_state_m = output_rnn_encoder
                final_state_encoder = [final_state_m, final_state_m]

           # Secuencia de salida de la ultima capa del encoder
            attention_states = output_encoder
            # Se usa el estado final del paso forward cmo estado inicial del decoder
            initial_state_decoder = final_state_encoder

        logger.info(f'-- initial_state_decoder: {initial_state_decoder}')
        logger.info(f'-- attention_states: {attention_states}')

        variable_summaries(initial_state_decoder, 'initial_state_decoder')
        variable_summaries(attention_states, 'attention_states')



    ##DECODER
    with tf.name_scope('decoder') as scope:


        W_decoder = tf.Variable(tf.truncated_normal([FLAGS.dim_lstm, num_classes], stddev=0.1), name='W_decoder')
        b_decoder = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias_decoder')
        variable_summaries(W_decoder, 'W_decoder')

        def loop_function(prev, _):
            # The next input are a softmax of the previous input
            relu_prev = tf.nn.relu(tf.matmul(prev, W_decoder) + b_decoder)
            return tf.nn.softmax(relu_prev)



        if FLAGS.rnn_encoder_type == 'LSTM':
            cell_dec = tf.contrib.rnn.LSTMCell(
                FLAGS.dim_lstm,
                initializer='glorot_uniform'
            )
        else: # 'GRU':
            cell_dec = tf.contrib.rnn.GRUCell(
                FLAGS.dim_lstm,
                kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform'
            )
        cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob= 1 - FLAGS.dropout_value)
 


        def decoder(teacher_forcing_bool):
            loop_f = None if teacher_forcing_bool else loop_function
            reuse = None if teacher_forcing_bool else True
            with tf.variable_scope(
                tf.get_variable_scope(), reuse=reuse) as scope:
                dec_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                    input_decoder,
                    initial_state_decoder,
                    attention_states,
                    cell_dec,
                    num_heads=FLAGS.num_heads,
                    loop_function=loop_f
                )
            return dec_outputs

        # If teacher_forcing = False --> TEST: reuse the previous predicted output for the next output with the loop function
        # If teacher_forcing = Frue -->  TRAIN: use the real previous output to predict the next output
        dec_outputs = tf.cond(teacher_forcing, lambda: decoder(True), lambda: decoder(False))
        #logger.info('-- dec_outputs: ', dec_outputs)



    with tf.name_scope('outputs') as scope:
        dense_outputs = [tf.nn.relu(tf.matmul(dec_o, W_decoder) + b_decoder) for dec_o in dec_outputs]
        output_proba = tf.concat([tf.expand_dims(t,1) for t in dense_outputs], 1)
        variable_summaries(dense_outputs, 'dense_outputs')
        #Prediction probs
        output = tf.concat([tf.expand_dims(tf.nn.softmax(t),1) for t in dense_outputs], 1)
        #logger.info('-- output: ', output)


    #Loss
    with tf.name_scope('loss') as scope:
        regularized_list = [
            'convolutions/Variable:0',
            'convolutions/Variable_1:0',
            'convolutions/Variable_2:0',
            'decoder/W_decoder:0',
            'attention_decoder/AttnW_0:0',
            'attention_decoder/weights:0',
            'attention_decoder/Attention_0/weights:0',
            'attention_decoder/AttnOutputProjection/weights:0'
        ]
        loss = tf.contrib.legacy_seq2seq.sequence_loss(dense_outputs, targets, weights, name='seq2seq')

        #loss_regularized = loss + FLAGS.lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if tf_var.name in regularized_list)
        loss_regularized = loss + FLAGS.lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        loss_summary = tf.summary.scalar("loss", loss)
        loss_regularized_summary = tf.summary.scalar("loss_regularized", loss_regularized)


    #Lists of all trainable variables
    train_vars = tf.trainable_variables()

    #Trainer
    with tf.name_scope('trainer') as scope:
        starter_learning_rate = FLAGS.learning_rate

        '''
        global_step = tf.Variable(0, trainable=False)
        learning_rate_mdl = tf.train.exponential_decay(
            starter_learning_rate,
            global_step,
            FLAGS.exponential_decay_step,
            FLAGS.exponential_decay_rate,
            staircase=True
        )
        optimizer = tf.train.AdamOptimizer(
            learning_rate_mdl,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )

        train_vars = tf.trainable_variables()
        grads = tf.gradients(loss_regularized, train_vars)

        # Passing global_step to minimize() will increment it at each step.
        train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
        '''

        global_step = tf.Variable(0, trainable=False)
        learning_rate_mdl = tf.train.exponential_decay(
            starter_learning_rate,
            global_step,
            FLAGS.exponential_decay_step,
            FLAGS.exponential_decay_rate,
            staircase=True
        )
        learning_rate_mdl_summary = tf.summary.scalar("learning_rate_mdl", learning_rate_mdl)

        # Passing global_step to minimize() will increment it at each step.
        train_op = tf.train.AdamOptimizer(
            learning_rate_mdl,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        ).minimize(loss_regularized, global_step=global_step)
        #).minimize(loss, global_step=global_step)


    # Saver
    saver = tf.train.Saver(max_to_keep=200)

    # Summaries
    summaries_dir = FLAGS.experiment
    with tf.name_scope('summaries') as scope:
        merged = tf.summary.merge_all()

    # Add to collection
    tf.add_to_collection('input_image', input_image)
    tf.add_to_collection('input_slides_len', input_slides_len)
    tf.add_to_collection('input_word_chars', input_word_chars)
    tf.add_to_collection('input_targets', input_targets)
    tf.add_to_collection('input_weights', input_weights)
    tf.add_to_collection('output_proba', output_proba)
    tf.add_to_collection('output', output)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('is_training', is_training)
    tf.add_to_collection('teacher_forcing', teacher_forcing)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('merged', merged)

    for t in tf.trainable_variables():
        logger.info(t.name)


logger.info('MODEL CREATED!')



def decode_response(response_array, decoder_dict):
    '''
    Convert numeric codes to output sequence of text
    '''
    response_text = []
    for i in range(response_array.shape[0]):
        response_dec = [np.argmax(r) for r in response_array[i,:,:]]
        response_text += [decode_text(response_dec, decoder_dict)]
    return response_text






def evaluate_dataset(
    sess,
    img_list,
    img_l_list,
    target_list,
    target_l_list,
    filenames_list,
    batch_size=256,
    print_results=True):


    loss_cumm = []
    real = []
    predict = []
    filenames = []
    correct = 0
    num_cases = 0
    seq = batch_generator_epoch(
        img_list,
        img_l_list,
        target_list,
        target_l_list,
        filenames_list,
        batch_size=batch_size,
        seq_length=FLAGS.seq_decoder_len,
        slides_stride=FLAGS.slides_stride,
        x_slide_size=FLAGS.x_slide_size,
        num_classes=num_classes,
        augmentation=False
    )
    for (images_b, slides_len_b, word_chars_b, targets_b, weights_b, _, _, filenames_b) in seq:
        feed_dict = {
            input_image: images_b,
            input_slides_len: slides_len_b,
            input_word_chars: word_chars_b,
            input_targets: targets_b,
            input_weights: weights_b,
            teacher_forcing: False, #False in eval to use predictions instead of real. 
            is_training: False} 
        loss_step, out_step, _, current_lr = sess.run([loss, output, output_proba, learning_rate_mdl], feed_dict)
        loss_cumm += [loss_step]

        response_predict_text = decode_response(out_step, decoder_dict)
        for resp in range(len(out_step)):
            real += [decode_text(targets_b[resp], decoder_dict)[:-4]]
            predict += [response_predict_text[resp][:-4]]
            num_cases += 1
            if decode_text(targets_b[resp], decoder_dict) == response_predict_text[resp]:
                correct += 1
        filenames += filenames_b

    #Details dataframe
    details_df = pd.DataFrame([real, predict, filenames]).T
    details_df.columns = ['real', 'predict', 'filename']

    # Calculate wer cer
    wer, cer = calculate_wer_cer(real, predict)
    if print_results:
        logger.info(f'Results no lexicon. Loss: {np.mean(loss_cumm)} - num_correct: {correct} - pct_correct: {float(correct)/float(num_cases)}')
        logger.info(f'Results no lexicon. WER: {wer} - CER: {cer}')
        
    return wer, cer, np.mean(loss_cumm), feed_dict, details_df







def train_batch(batch_size, lr=0.001, epoch=1):
    '''
    Train step
    '''
    
    wer_val_list = []
    min_wer_val = 1
    continue_training = True
    current_step = 1

    while continue_training:
        logger.info(f'\n')
        logger.info(f'============================================')
        logger.info(f'TRAINING EPOCH: {epoch}')

        tic = time.clock()

        summary_data = []
        
        loss_cumm = []

        seq = batch_generator_epoch(
            img_trn,
            img_l_trn,
            target_trn,
            target_l_trn, 
            filenames_list_trn,
            batch_size=FLAGS.batch_size,
            seq_length=FLAGS.seq_decoder_len,
            slides_stride=FLAGS.slides_stride,
            x_slide_size=FLAGS.x_slide_size,
            num_classes=num_classes,
            augmentation=FLAGS.data_augmentation
        )
        for s in seq:
            feed_dict = {
                input_image: s[0],
                input_slides_len: s[1],
                input_word_chars: s[2],
                input_targets: s[3],
                input_weights: s[4],
                teacher_forcing: FLAGS.teacher_forcing,  #Could be True in training to use real instead of predictions. teacher forcing
                is_training: True 
            }
            _, loss_t = sess.run([train_op, loss], feed_dict)
            loss_cumm += [loss_t]

        # Sumaries train 
        summary_str= sess.run(merged, feed_dict)
        train_tf_writer.add_summary(summary_str, epoch)

        # Save model
        savefile = saver.save(sess, FLAGS.experiment +'/model-epoch', global_step=epoch)
        logger.info(f'Model saved in {savefile}')

        logger.info(f'Time train: {time.clock()-tic}')



        #####################################
        # Evaluate train / val / test whit feed_previous = False
        #####################################

        #Train
        tic = time.clock()
        logger.info(f'---------------------------------------------')
        logger.info(f'Dataset: TRAIN')
        wer_trn, cer_trn, loss_trn, feed_dict, details_trn_df = evaluate_dataset(
            sess,
            img_trn,
            img_l_trn,
            target_trn,
            target_l_trn,
            filenames_list_trn
        )
        details_trn_df['partition'] = 'train'
        summary_data += [(epoch, 'train', wer_trn, cer_trn, loss_trn)]
        logger.info(f'Time to evaluate train: {time.clock()-tic}')
        # Sumaries train
        summary_str = sess.run(merged, feed_dict)
        train_writer.add_summary(summary_str, epoch)

        #Validation
        tic = time.clock()
        logger.info(f'---------------------------------------------')
        logger.info(f'Dataset: VALID')
        wer_val, cer_val, loss_val, feed_dict, details_val_df = evaluate_dataset(
            sess,
            img_val,
            img_l_val,
            target_val,
            target_l_val,
            filenames_list_val
        )
        details_val_df['partition'] = 'valid'
        summary_data += [(epoch, 'valid', wer_val, cer_val, loss_val)]
        logger.info(f'Time to evaluate validation: {time.clock()-tic}')
        # Sumaries valid
        summary_str = sess.run(merged, feed_dict)
        validation_writer.add_summary(summary_str, epoch)

        #Test
        tic = time.clock()
        logger.info(f'---------------------------------------------')
        logger.info(f'Dataset: TEST')
        wer_tst, cer_tst, loss_tst, feed_dict, details_tst_df = evaluate_dataset(
            sess,
            img_tst,
            img_l_tst,
            target_tst,
            target_l_tst,
            filenames_list_tst
        )
        details_tst_df['partition'] = 'test'
        summary_data += [(epoch, 'test', wer_tst, cer_tst, loss_tst)]
        logger.info(f'Time to evaluate test: {time.clock()-tic}')


        # Sumaries test
        summary_str = sess.run(merged, feed_dict)
        test_writer.add_summary(summary_str, epoch)

        # Creation of details dataframe to the epoch
        details_df = pd.concat([details_trn_df, details_val_df, details_tst_df])
        details_df['epoch'] = epoch
        details_df[['epoch', 'partition', 'filename', 'real', 'predict']].to_csv(os.path.join(FLAGS.experiment, f'details_df_{epoch}.csv'), index=False)

        #update of the summary dataframe to the epoch
        summary_df = pd.DataFrame(
            summary_data,
            columns=['epoch', 'partition', 'wer',' cer', 'loss']
        )
        summary_df.to_csv(os.path.join(FLAGS.experiment, f'summary_df.csv'), index=False, mode='a', header=False)

        #####################################



        # Early stoping criteria
        # Criteria with standard lexicon
        #wer, cer = evaluate_corpus(real_tst, predict_tst, corpus_dict, print_results=False)
        #logger.info(f'VALIDATION LEXICON - WER: {wer} - CER: {cer}')
        wer_val_list += [wer_val]
        if min_wer_val > wer_val:
            min_wer_val = wer_val
            # Save best model
            savefile = saver.save(sess, FLAGS.experiment +'/best_model')
            logger.info(f'Model saved in {savefile}')
        if current_step > FLAGS.min_steps:
            # Stopping criteria: not improvement in last 10 epochs
            if min_wer_val < np.min(wer_val_list[-FLAGS.early_stopping_steps:]):
                logger.info(f'Stopping training: No improvement of wer in val in 10 epochs.')
                logger.info(f'min wer val: {min_wer_val} - wer val last 10 epochs: {wer_val_list[-10:]} \n')
                continue_training = False

        if current_step > FLAGS.max_steps:
            logger.info(f'Stopping training: Max steps {FLAGS.max_steps} reached.\n')
            continue_training = False

        epoch += 1
        current_step += 1

        #sync to s3
        #os.system('aws s3 sync '+FLAGS.experiment+' s3://tesis-sueiras/'+FLAGS.experiment.split('/')[-1])





# Load lexicon to early stopping
#corpus_dict = pickle.load(open(FLAGS.dictionary_pickle_path, "rb"))


with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Merge all the summaries and write them out to /tmp/mnist_logs
    train_tf_writer = tf.summary.FileWriter(summaries_dir + '/train_tf', sess.graph)
    train_writer = tf.summary.FileWriter(summaries_dir + '/train')
    validation_writer = tf.summary.FileWriter(summaries_dir + '/val')
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')


    # Initialize vars if dont exist previous checkpoints.
    ckpt = tf.train.get_checkpoint_state(FLAGS.experiment)
    if ckpt == None:
        # Initialize vars
        tf.global_variables_initializer().run()
        logger.info('vars initialized!')
        epoch_ini = 1
    else:
        # Load best model
        saver.restore(sess, ckpt.model_checkpoint_path)
        if os.path.basename(ckpt.model_checkpoint_path).split('-')[-1] == 'best_model':
            epoch_ini = 1
        else:
            epoch_ini = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[-1]) + 1
        logger.info(f'model loaded: {ckpt.model_checkpoint_path}')

    # Train model
    train_batch(FLAGS.batch_size, lr=FLAGS.learning_rate, epoch=epoch_ini)






# Accuracy measures of the best model over train, valid and test data
logger.info('=============================')
logger.info('       BEST MODEL RESULTS    ')
logger.info('=============================\n')

logger.info(f'Stardard lexicon: {FLAGS.dictionary_pickle_path}')
standard_lexicon_dict = pickle.load(open(FLAGS.dictionary_pickle_path, "rb" ))
logger.info(f'Standard lexicon size: {len(standard_lexicon_dict)}')

#test_lexicon_dict = get_real_test_dictionary_set(FLAGS.decoder_dict_data)
#logger.info(f'Test lexicon size: {len(test_lexicon_dict)}')


with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #Restore best model
    saver.restore(sess, FLAGS.experiment +'/best_model')
    
    summary_data = []

    #Train
    wer_none_trn, cer_none_trn, loss_trn, _, details_trn_df = evaluate_dataset(
            sess,
            img_trn,
            img_l_trn,
            target_trn,
            target_l_trn,
            filenames_list_trn,
            print_results=False
    )
    summary_data += [('train', 'no_lexicon', wer_none_trn, cer_none_trn)]
    logger.info(f'TRAIN. No  lexicon. WER: {wer_none_trn} - CER: {cer_none_trn}  -Loss: {loss_trn}')
    details_trn_df['partition'] = 'train'
    real_trn = list(details_trn_df['real'].values)
    predict_trn = list(details_trn_df['predict'].values)

    wer_std_trn, cer_std_trn, predict_std_trn = evaluate_corpus(real_trn, predict_trn, standard_lexicon_dict)
    summary_data += [('train', 'std_lexicon', wer_std_trn, cer_std_trn)]
    logger.info(f'TRAIN. Std lexicon. WER: {wer_std_trn} - CER: {cer_std_trn}')
    details_trn_df['predict_std_lexicon'] = predict_std_trn

    #wer_tst_trn, cer_tst_trn, predict_tst_trn = evaluate_corpus(real_trn, predict_trn, set(real_trn))
    #summary_data += [('train', 'tst_lexicon', wer_tst_trn, cer_tst_trn)]
    #logger.info(f'TRAIN. Train lexicon. WER: {wer_tst_trn} - CER: {cer_tst_trn}')
    #details_trn_df['predict_tst_lexicon'] = predict_tst_trn


    #Validation
    wer_none_val, cer_none_val, loss_val, _, details_val_df = evaluate_dataset(
            sess,
            img_val,
            img_l_val,
            target_val,
            target_l_val,
            filenames_list_val,
            print_results=False
    )
    summary_data += [('valid', 'no_lexicon', wer_none_val, cer_none_val)]
    logger.info(f'VALID. No  lexicon. WER: {wer_none_val} - CER: {cer_none_val}  -Loss: {loss_val}')
    details_val_df['partition'] = 'valid'
    real_val = list(details_val_df['real'].values)
    predict_val = list(details_val_df['predict'].values)

    wer_std_val, cer_std_val, predict_std_val = evaluate_corpus(real_val, predict_val, standard_lexicon_dict)
    summary_data += [('valid', 'std_lexicon', wer_std_val, cer_std_val)]
    logger.info(f'VALID. Std lexicon. WER: {wer_std_val} - CER: {cer_std_val}')
    details_val_df['predict_std_lexicon'] = predict_std_val

    #, cer_tst_val, predict_tst_val = evaluate_corpus(real_val, predict_val, set(real_val))
    #summary_data += [('valid', 'tst_lexicon', wer_tst_val, cer_tst_val)]
    #logger.info(f'VALID. Valid lexicon. WER: {wer_tst_val} - CER: {cer_tst_val}')
    #details_val_df['predict_tst_lexicon'] = predict_tst_val


    #Test
    wer_none_tst, cer_none_tst, loss_tst, _, details_tst_df = evaluate_dataset(
            sess,
            img_tst,
            img_l_tst,
            target_tst,
            target_l_tst,
            filenames_list_tst,
            print_results=False
    )
    summary_data += [('test', 'no_lexicon', wer_none_tst, cer_none_tst)]
    logger.info(f'TEST . No  lexicon. WER: {wer_none_tst} - CER: {cer_none_tst}  -Loss: {loss_tst}')
    details_tst_df['partition'] = 'test'
    real_tst = list(details_tst_df['real'].values)
    predict_tst = list(details_tst_df['predict'].values)

    wer_std_tst, cer_std_tst, predict_std_tst = evaluate_corpus(real_tst, predict_tst, standard_lexicon_dict)
    summary_data += [('test', 'std_lexicon', wer_std_tst, cer_std_tst)]
    logger.info(f'TEST . Std lexicon. WER: {wer_std_tst} - CER: {cer_std_tst}')
    details_tst_df['predict_std_lexicon'] = predict_std_tst

    wer_tst_tst, cer_tst_tst, predict_tst_tst = evaluate_corpus(real_tst, predict_tst, set(real_tst))
    summary_data += [('test', 'tst_lexicon', wer_tst_tst, cer_tst_tst)]
    logger.info(f'TEST . Tst lexicon. WER: {wer_tst_tst} - CER: {cer_tst_tst}')
    details_tst_df['predict_tst_lexicon'] = predict_tst_tst

    logger.info(f'Test lexicon size: {len(set(real_tst))}')


    # Creation of details dataframe to the best model
    details_df = pd.concat([details_trn_df, details_val_df, details_tst_df])
    details_df[['partition', 'filename', 'real', 'predict', 'predict_std_lexicon', 'predict_tst_lexicon']].to_csv(
        os.path.join(FLAGS.experiment, f'details_df_best_model.csv'), index=False
    )


    #Creation of summary dataframe to the best model
    summary_df = pd.DataFrame(
        summary_data,
        columns=['partition', 'lexicon', 'wer',' cer']
    )
    summary_df.to_csv(os.path.join(FLAGS.experiment, f'summary_df_best_model.csv'), index=False)


    #sync to s3
    #os.system('aws s3 sync '+FLAGS.experiment+' s3://tesis-sueiras/'+FLAGS.experiment.split('/')[-1])

    logger.info('Done!')


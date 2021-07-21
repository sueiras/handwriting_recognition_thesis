# datasets.py
# Functions to manage handwritting datasets

import os
import codecs
import cv2
import random
import math
import logging

import tensorflow as tf
import numpy as np

from PIL import Image

#import matplotlib as mpl
#mpl.use('TkAgg')  # if not, error of core dump
#from matplotlib import pyplot as plt



def encode_target_hex(target):
    ''' Encode text from ascii to hex
    '''
    return codecs.encode(target.encode('utf-8'), "hex").decode('ascii')



def decode_target_hex(target_encoded):
    ''' Decode text from hex to ascii
    '''
    return codecs.decode(target_encoded, "hex").decode('utf-8')



def create_encoder_dir_old():
    char_list = []
    for c in range(32, 127):
        char_list += [chr(c)]

    #RIMES extra characters
    char_list += ['²','°','ù','é','ê','ô','è','É','î','à','ë','û','â','ç','ï']

    #Osborne extra characters
    char_list += ['¡','¿','¡','©','­','±','³','º']

    # Spanish extra characters
    char_list += ['Ñ','ñ','á','é','í','ó','ú','Á','É','Í','Ó','Ú','ü','Ü','¿','¡','ª','º']

    # French extra characters
    '''char_list += ['À','à','Â','â','Æ','æ','Ç','ç','È','è','É','é','Ê','ê','Ë','ë',
              'Î','î','Ï','ï','Ô','ô','Œ','œ','Ù','ù','Û','û','Ü','ü','«','»','€']
    '''
    char_list =sorted(list(set(char_list)))

    encode_target_dict = {}
    for i, c in enumerate(char_list):
        encode_target_dict[c] = i

    return encode_target_dict



def create_encoder_dir():
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

    encode_target_dict = {}
    for i, c in enumerate(char_list):
        encode_target_dict[chr(c)] = i

    return encode_target_dict

#encode_target_dict = create_encoder_dir()



def adjust_image_old(img_file, x_size=192, y_size=48, x_spaces_ini=0):
    ''' Read image and adjust to a fixed x and y size
    '''
    im = Image.open(img_file).convert('L') # read and convert to grayscale

    if np.max(im)>0:
        x, y = im.size
        factor = y_size/y
        new_x = min( max(1, int(factor*x)), x_size-x_spaces_ini)

        img_resize_array = np.asarray(im.resize((new_x, y_size)))

        img_adjusted = np.concatenate([np.zeros((y_size, x_spaces_ini)), img_resize_array], axis=1)

        new_x_size = img_adjusted.shape[1]
        if new_x_size < x_size:
            img_adjusted = np.concatenate([img_adjusted, np.zeros((y_size, x_size-new_x_size))], axis=1)

        if np.max(img_adjusted)>0:
            return img_adjusted
        else:
            return []
    else:
        return []



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
        
    return img/img.max()




def read_word_image(path, f, encoder_dict, x_size=192, y_size=48, target_max_size=19, x_spaces_ini=0):
    ''' Read image from file and get the target from the image name

    '''
    # adjust and resize to the final size
    img_adjusted = adjust_image(os.path.join(path, f), x_size=x_size, y_size=y_size, x_spaces_ini=x_spaces_ini)

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




# Image augmentation code
# ========================
def move_img(img):
    pixels_move = 1 + int(random.random()*10)
    img2 = np.ones_like(img)*0
    img2[:,pixels_move:] = img[:,:-pixels_move]
    return img2

def resize_down(img):
    factor = 0.95 - random.random()/4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img)*0
    img2[(h_ini-h_fin)//2:-(h_ini-h_fin)//2, :w_fin] = img1
    return img2

def resize_up(img):
    factor = 1 + random.random()/4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = img1[h_fin-h_ini:, :w_ini]
    return img2


def get_img_augmented(image_list, augment=True):
    if augment:
        augmented_image_list = []
        for img in image_list:
            if len(img.shape)>2:
                img = img[:,:,0]

            # Move left
            img = move_img(img)

            # Skew
            if random.random() < 0.8 :
                shape_ini = img.shape
                angle = (random.random()-0.5)/3.
                M = np.float32([[1, -angle, 0.5*img.shape[0]*angle], [0, 1, 0]])
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                     flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
            #Resize
            if random.random() < 0.4:
                img = resize_down(img)
            elif random.random() < 0.4:
                img = resize_up(img)

            #Erode - dilate
            if random.random() < 0.3:
                img = cv2.erode(img, np.ones(2, np.uint8), iterations=1)
            elif random.random() < 0.3:
                img = cv2.dilate(img, np.ones(2, np.uint8), iterations=1)

            augmented_image_list += [img]


    else: # Not augmentation
        augmented_image_list = image_list

    return np.array(augmented_image_list)






# TFRecords functions
#====================

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def image_to_tfexample(image_data, image_length, height, width, target_seq, label_length):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/length': int64_feature(image_length),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/label': int64_feature(target_seq),
        'image/label_length': int64_feature(label_length),
    }))


def _add_to_tfrecord(image, target, image_len, target_len, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.
    Args:
      data_filename: The filename of the MNIST images.
      labels_filename: The filename of the MNIST labels.
      num_images: The number of images in the dataset.
      tfrecord_writer: The TFRecord writer to use for writing.
    """

    with tf.Graph().as_default():
        img = tf.placeholder(dtype=tf.uint8, shape=(image.shape))
        img_expand = tf.expand_dims(img, -1)
        encoded_png = tf.image.encode_png(img_expand)

        with tf.Session('') as sess:
            png_string = sess.run(encoded_png, feed_dict={img: image})
            example = image_to_tfexample(png_string, image_len, image.shape[0],
                                         image.shape[1], target, target_len)
            tfrecord_writer.write(example.SerializeToString())





def generate_tfrecod_word(word_images_path,
                          num_images,
                          tfrecords_path,
                          tfname='iam_',
                          num_shards=10,
                          x_size=192,
                          y_size=48,
                          target_max_size=19,
                          max_word_size=0):
    ''' Generate the TFRecords based on the images files of the handwritten words
    '''
    if not tf.gfile.Exists(tfrecords_path):
        tf.gfile.MakeDirs(tfrecords_path)

    encoder_dict = create_encoder_dir()

    # List of word images
    list_word_images_all = [f for f in os.listdir(word_images_path) if os.path.isfile(os.path.join(word_images_path, f))]
    np.random.shuffle(list_word_images_all)
    list_word_images_all = list_word_images_all[:min(num_images, len(list_word_images_all))]

    print(list_word_images_all[0])

    # Words of the selected length
    if max_word_size==0: # All the words
        list_word_images = [os.path.join(word_images_path, f) for f in list_word_images_all]

    else: # only words of selected length
        list_word_images = []
        for f in list_word_images_all:
            hex_name = f.split('.')[0].split('_')[-1] # extract hexadecimal part
            target_c = decode_target_hex(hex_name) # convert to string
            l = len(target_c)
            #print(f, l)
            if l <=max_word_size:
                list_word_images += [os.path.join(word_images_path, f)]


    # Calculate num images final
    num_images_final = min(len(list_word_images), num_images)
    print(len(list_word_images), num_images)
    if num_images_final < num_images:
        logging.info('Using %s images of a total of %s images', num_images_final, num_images)



    # Records per shard
    num_per_shard = int(math.ceil(num_images_final / num_shards))


    tfrecords_files_list = []
    for shard_id in range(num_shards):

        # Create new tfrecord file
        tfrecord_filename = os.path.join(tfrecords_path, tfname + str(shard_id).zfill(4)+'.tfrecord')
        tfrecords_files_list += [tfrecord_filename]

        with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:

            # Identify start and end index of each shard
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, num_images_final)
            for i in range(start_ndx, end_ndx):
                if i%100==0:
                    logging.info('Shard %s, image num %s of %s.', shard_id, i, num_images_final)

                image, target, image_len, target_len = read_word_image(
                    word_images_path,
                    list_word_images[i],
                    encoder_dict,
                    x_size=x_size,
                    y_size=y_size,
                    target_max_size=target_max_size
                )
                
                if image != []:
                    _add_to_tfrecord(image, target, image_len, target_len, tfrecord_writer)
                else:
                    logging.info('Error in image %s', list_word_images[i])

    return tfrecords_files_list





# Template function and doc
#def f():
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """

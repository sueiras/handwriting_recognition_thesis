#! /usr/bin/env python

# imports
import codecs
import datetime
import fnmatch
import glob
import logging
import os
import shutil
import sys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def encode_target_hex(target):
    ''' Encode text from ascii to hex
    '''
    return codecs.encode(target.encode('utf-8'), "hex").decode('ascii')


def decode_target_hex(target_encoded):
    ''' Decode text from hex to ascii
    '''
    return codecs.decode(target_encoded, "hex").decode('utf-8')


# Generator of list of files in a folder and subfolders
def gen_find(filepat,top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist,filepat):
            yield os.path.join(path,name)



#mnist database
def get_mnist():
    """
    Return mnist normalized in [0,1]
    """

    mnist = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    id = np.array(range(len(mnist[0][0])))
    X = np.array(mnist[0][0], dtype=np.float32)
    X = X/255
    y = np.array(mnist[0][1], dtype=np.uint8)
    id_trn, id_val, X_trn, X_val, y_trn, y_val = train_test_split(id, X, y, train_size=50000, random_state=42)

    id_tst = np.array(range(len(mnist[1][0])))
    X_tst = np.array(mnist[1][0], dtype=np.float32)
    X_tst = X_tst/255
    y_tst = np.array(mnist[1][1], dtype=np.uint8)

    y_labels_dict = {i:str(i) for i in [0,1,2,3,4,5,6,7,8,9]}
    
    return id_trn, np.expand_dims(X_trn, axis=-1), y_trn, id_val, np.expand_dims(X_val, axis=-1), y_val, id_tst, np.expand_dims(X_tst, axis=-1), y_tst, y_labels_dict







#Load TICH

def read_tich_data(filelist, target_dict, size=0):
    """
    """
    
    id=[]
    X=[]
    y=[]
    for fname in filelist:
        file = fname.split('/')[-1]
        id += [file]

        img = Image.open(fname)
        #resize
        if size != 0:
            img = img.resize((size, size))
        X += [np.array(img)/255]

        y += [target_dict[file]]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint16)
    return id, X, y

def get_target_dict(data_path, filename):
    """
    """
    data_df = pd.read_csv(
        os.path.join(data_path,'TICH', 'extract', filename), delimiter=' ', names=['file', 'y']
        )
    data_df['filename']  = data_df['file'].str.split('/').str[-1]
    data_df.set_index('filename', inplace=True)
    data_dict = data_df.to_dict()
    target_dict = data_dict['y']
    return target_dict

def get_TICH(data_path, size=None):
    """
    """
    # Get decoder target dict
    y_label = pd.read_csv(
            os.path.join(data_path,'TICH','extract','labels.txt'), names=['y_label'])
    y_labels_dict = y_label.to_dict()['y_label']

    #Read train
    train_target_dict = get_target_dict(data_path, 'train.txt')

    # Generate train
    filelist = glob.glob(os.path.join(data_path,'TICH','extract','imgs','train*.png'))
    id, X, y = read_tich_data(
        filelist, 
        train_target_dict,
        size=size
        )

    # Split train in train and valid
    id_trn, id_val, X_trn, X_val, y_trn, y_val = train_test_split(id, X, y, train_size=0.95, random_state=42)

    # read test
    test_target_dict = get_target_dict(data_path, 'valid.txt')

    # Generate test
    filelist = glob.glob(os.path.join(data_path,'TICH','extract','imgs','valid*.png'))
    id_tst, X_tst, y_tst = read_tich_data(
        filelist,
        test_target_dict,
        size=size
        )

    return id_trn, np.expand_dims(X_trn, axis=-1), y_trn, id_val, np.expand_dims(X_val, axis=-1), y_val, id_tst, np.expand_dims(X_tst, axis=-1), y_tst, y_labels_dict






#Read NIST data
def read_NIPS_partition(path, partition, char_list, encode_y, size=0):
    """
    """
    id = []
    X = []
    y = []
    for char in char_list:
        letter = decode_target_hex(char)
        if partition=='trn':
            images_list = gen_find("*.png", path + char + '/train_'+char) 
        else:
            images_list = gen_find("*.png", path + char + '/hsf_4/') 

        for img_name in images_list:
            img = Image.open(img_name)

            #crop black borders. Are too big
            img = img.crop((32, 32, 96, 96))

            #resize
            if size != 0:
                img = img.resize((size, size))
            im_array = np.array(img)[:,:,0]
            im_array = im_array - np.min(im_array)
            im_array = im_array/np.max(im_array)

            id +=[img_name.split("/")[-1]]
            X += [im_array]
            y += [encode_y[letter]]

    X = 1. - np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint16)

    return id, X, y


def get_NIST(path, char_list, size=0):
    """
    """
    #Encoder decoder target dict
    decode_y={}
    encode_y={}
    char_list_hex = []
    for i , char in enumerate(char_list):
        char_list_hex += [encode_target_hex(char)]
        decode_y[i] = char
        encode_y[char] = i

    # Generate train
    id, X, y = read_NIPS_partition(path, 'trn', char_list_hex, encode_y, size=size)

    # Split train in train and valid
    id_trn, id_val, X_trn, X_val, y_trn, y_val = train_test_split(id, X, y, train_size=0.9, random_state=42)

    # Generate test
    id_tst, X_tst, y_tst = read_NIPS_partition(path, 'tst', char_list_hex, encode_y, size=size)

    return id_trn, np.expand_dims(X_trn, axis=-1), y_trn, id_val, np.expand_dims(X_val, axis=-1), y_val, id_tst, np.expand_dims(X_tst, axis=-1), y_tst, decode_y





#Read unipen online data
def read_unipen_data(data_path, partition_file, encode_target, char_list=[], size=0):
    """
    """
    with open(data_path + '/split/' + partition_file) as f:
        list_files = f.read().splitlines()

    id=[]
    X=[]
    y=[]
    for f in list_files:
        target = chr(int(f.split('/')[0]))
        if target in char_list:
            id += [f]
            im = Image.open(data_path + '/curated/' + f)
            if size != 0:
                im = im.resize((size, size))
            X += [np.array(im)/255]
            y += [encode_target[target]]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint16)
    
    return id, X, y


def get_unipen(data_path, char_list=[], size=0):
    """
    """

    if char_list == []:
        with open(data_path + '/split/trn.txt') as file:
            list_files = file.read().splitlines()
        target_list = [chr(int(f.split('/')[0])) for f in list_files]
        char_list = sorted(list(set(target_list)))    


    # target dictionaries
    y_labels_dict = {}
    encode_target = {}
    for i , c in enumerate(char_list):
        y_labels_dict[i] = c
        encode_target[c] = i
        
    id_trn, X_trn, y_trn = read_unipen_data(
        data_path,
        'trn.txt',
        encode_target,
        char_list=char_list,
        size=size
    )
    id_val, X_val, y_val = read_unipen_data(
        data_path,
        'val.txt',
        encode_target,
        char_list=char_list,
        size=size
    )
    id_tst, X_tst, y_tst = read_unipen_data(
        data_path,
        'tst.txt',
        encode_target,
        char_list=char_list,
        size=size
    ) 
        
    return id_trn, np.expand_dims(X_trn, axis=-1), y_trn, id_val, np.expand_dims(X_val, axis=-1), y_val, id_tst, np.expand_dims(X_tst, axis=-1), y_tst, y_labels_dict







# Functions to augment the new unipen offline database
#------------------------------------------------------



def move_characters(X, y, y_labels_dict):
    """
    - New characters concatenating in the border of the character the start and/or end of other characters
    - new characters moving up /down characters (rescale and place in the selected positions). Classes:
        - Medium or high or low: a, c, e, i, m, n, o, r, s, u, v, w, x, z
        - Low - Medium: g, j, p, q, y 
        - Medium-High: b, d, f, h, k, l, t
    - Two ways to concatenate: with separation or withow separation (the border are aprox 4pixels)  
    - concatenate original characters with original characters and moved characters with moved characters 
    """
    X_moved=[]
    y_moved=[]
    for i,y_value in enumerate(y):
        y_char = y_labels_dict[y_value]
        if y_char in ['a', 'c', 'e', 'i', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z']:
            #Resize to 40x40 and place botton, medium and top 
            img1 = cv2.resize(
                src=X[i,:,:],
                dsize=(46,46),
                interpolation=cv2.INTER_CUBIC
            )

            
            img_center = np.zeros((64,64))
            img_center[9:55, 9:55] = img1 
            X_moved += [img_center]
            y_moved += [y_value]
            
            img_top = np.zeros((64,64))
            img_top[:46, 9:55] = img1 
            X_moved += [img_top]
            y_moved += [y_value]

            img_botton = np.zeros((64,64))
            img_botton[18:, 9:55] = img1 
            X_moved += [img_botton]
            y_moved += [y_value]

        elif y_char in ['g', 'j', 'p', 'q', 'y']:
            #Resize to 52x52 and place botton
            img1 = cv2.resize(
                src=X[i,:,:],
                dsize=(46,46),
                interpolation=cv2.INTER_CUBIC
            )
            img_botton = np.zeros((64,64))
            img_botton[18:, 9:55] = img1 
            X_moved += [img_botton]
            y_moved += [y_value]

        elif y_char in ['b', 'd', 'f', 'h', 'k', 'l', 't', 'A', 'B', 'C', 'D', 'E',
                        'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
                        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            #Resize to 52x52 and place on top
            img1 = cv2.resize(
                src=X[i,:,:],
                dsize=(46,46),
                interpolation=cv2.INTER_CUBIC
            )
            img_top = np.zeros((64,64))
            img_top[:46, 9:55] = img1 
            X_moved += [img_top]
            y_moved += [y_value]

    return np.array(X_moved, dtype=np.float32), np.array(y_moved, dtype=np.uint16)




def get_borders(img, treshold=0.):
    """
    Create 64x8 borders left and rigth
    - Include in the border of each character the start/end of other letters 1 and 3 pixels
    - Create borders: select random characters and identify an cut borders of 2 and 4 pixels in each side
    - For each character select a random border and generate new character. Dont move the position of the original character
    - For each character identify how many need to generate to obtain a balanced sample of 1000
    """
    #Identify border of the letter
    border_left = img[:,:8]
    left_position = 0
    for i in range(0,56,1):
        if np.max(img[:,i]) > treshold:
            left_position = i
            break
    rigth_position = 64
    border_rigth = img[:,56:]
    for i in range(63,8,-1):
        if np.max(img[:,i]) > treshold:
            rigth_position = i
            break
    return img[:,left_position:left_position+8], img[:,rigth_position-8:rigth_position], img[:,left_position:rigth_position]




def generate_borders_images(X, y, rigth_borders, left_borders, multiplier_sample=4):
    """
    Join the borders to the original images
    """
    X_left = []
    y_left = []
    X_rigth = []
    y_rigth = []
    for i,y_value in enumerate(y):
        for j in range(multiplier_sample//2): # two images each iteration
            n_borders =  int(random.uniform(0, X.shape[0]))
            img_left = np.copy(X[i,:,:])
            img_left[:,:5] = rigth_borders[n_borders,:,-5:] #put the rigth side of the rigth border
            X_left += [img_left]
            y_left += [y_value]
            
            img_rigth = np.copy(X[i,:,:])
            img_rigth[:,-5:] = left_borders[n_borders,:,:5] #put the left side of the left border
            X_rigth += [img_rigth]
            y_rigth += [y_value]
            
    X_out  = np.array(X_rigth + X_left, dtype=np.float32)
    y_out  = np.array(y_rigth + y_left)

    return X_out, y_out




def transform_dataset(
    X, y, y_labels_dict,
    multiplier_sample=2,
    output_path = '/tmp/transform_dataset_test_',
):
    """
    """

    # Move characters
    X_moved, y_moved = move_characters(X, y, y_labels_dict)

    # Get borders
    left_borders = []
    rigth_borders = []
    img_no_borders = []
    for img in X_moved:
        l, r, _ = get_borders(img, treshold=0.)
        left_borders += [l]
        rigth_borders += [r]
    left_borders = np.array(left_borders, dtype=np.float32)
    rigth_borders = np.array(rigth_borders, dtype=np.float32)

    # Add borders
    X_augmented, y_augmented = generate_borders_images(
        X_moved, y_moved, rigth_borders, left_borders, multiplier_sample=multiplier_sample
        )

    # standarize 
    X_augmented = X_augmented/np.max(X_augmented)

    #shuffle
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=42)

    X_augmented = np.expand_dims(X_augmented, axis=-1)

    # Save as numpy arays
    np.save(output_path + 'X.npy', X_augmented)
    np.save(output_path + 'y.npy', y_augmented)
    
    return X_augmented, y_augmented
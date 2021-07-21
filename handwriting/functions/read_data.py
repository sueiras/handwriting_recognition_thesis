# read_data.py
# Functions to read handwritting datasets

import os
import codecs
import cv2
import random
import math
import logging
import glob

import numpy as np

from PIL import Image

#import matplotlib as mpl
#mpl.use('TkAgg')  # if not, error of core dump
from matplotlib import pyplot as plt


# create logger
module_logger = logging.getLogger('read_data.library')




def encode_target_hex(target):
    ''' Encode text from ascii to hex
    '''
    return codecs.encode(target.encode('utf-8'), "hex").decode('ascii')



def decode_target_hex(target_encoded):
    ''' Decode text from hex to ascii
    '''
    return codecs.decode(target_encoded, "hex").decode('utf-8')




def create_encoder_dir():
    ''' Common encoder dir to rimes iam and osborne
    '''
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
#create_encoder_dir()







def adjust_image(img_file, x_size=192, y_size=48, x_blanks_ini=5):
    ''' Read image and adjust to a fixed x and y size
    Inputs:
        img_file
        x_size=192
        y_size=48
        x_blanks_ini=5
    '''
    im = Image.open(img_file).convert('L') # read and convert to grayscale
    
    if np.max(im)>0:
        x, y = im.size
        factor = y_size/y
        new_x = min( max(1, int(factor*x)), x_size-x_blanks_ini)
        
        img_resize_array = np.asarray(im.resize((new_x, y_size)))
        
        img_adjusted = np.concatenate([np.zeros((y_size, x_blanks_ini)), img_resize_array], axis=1)
                                       
        new_x_size = img_adjusted.shape[1]
        if new_x_size < x_size:
            img_adjusted = np.concatenate([img_adjusted, np.zeros((y_size, x_size-new_x_size))], axis=1)

        if np.max(img_adjusted)>0:
            return img_adjusted
        else:
            module_logger.debug('Image adjusted is 0 in adjust_image method. Filename: %s', img_file)
            return []
    else:
        module_logger.debug('Image original is 0 in adjust_image method. Filename: %s', img_file)
        return []



def read_word_image(filename, get_target=True, encoder_dict={}, x_size=192, y_size=48, target_max_size=19, x_blanks_ini=5):
    ''' Read image from file and get the target from the image name
    Inputs:
        filename
        get_target=True
        encoder_dict={}
        x_size=192
        y_size=48
        target_max_size=19
    '''
    # adjust and resize to the final size
    img_adjusted = adjust_image(filename, x_size=x_size, y_size=y_size, x_blanks_ini=x_blanks_ini)
    
    if img_adjusted != []:
        #Calculate image_len
        image_len = np.max(np.nonzero(np.max(img_adjusted, axis=0))) 
            
        # Target
        if get_target:
            f = os.path.basename(filename)
            file_name = '.'.join(f.split('.')[:-1])
            hex_name = file_name.split('_')[-1] # extract hexadecimal part
            target_c = decode_target_hex(hex_name) # convert to string
            target_ini = [encoder_dict[k] for k in target_c] # encode to 
            if len(target_ini)>target_max_size: # Pendiente de resolver mejor
                target_ini = target_ini[:target_max_size]
            target_len = len(target_ini)
            target = np.ones([target_max_size], dtype=np.uint8)*(-1)
            target[:target_len] = target_ini
        else:
            target = ''
            target_len = 0
        return img_adjusted, list(target), image_len, target_len
    else:
        module_logger.debug('Error adjusting image with adjust_image. Filename: %s', filename)
        return [], '', 0, 0



    
# Image augmentation code
# ========================
def move_img(img):
    pixels_move = 1 + int(random.random()*10)
    img2 = np.ones_like(img)*0
    img2[:,pixels_move:] = img[:,:-pixels_move]
    return img2

def resize_down(img):
    factor = 0.95 - random.random()/4.
    h_ini, _ = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img)*0
    img2[(h_ini-h_fin)//2:-(h_ini-h_fin)//2, :w_fin] = img1
    return img2

def resize_up(img):
    factor = 1 + random.random()/4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, _ = img1.shape
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
        
    # convert negatives to 0
    augmented_image_list = np.clip(augmented_image_list, a_min=0, a_max=None)
    
    #rescale to [0,1]
    augmented_image_list = [img/np.max(img) for img in augmented_image_list]
    
        
    return np.array(augmented_image_list, dtype=np.float32)




def decode_text(text_array, decoder_dict, eol_code=0):
    '''Decode the target from numbers to words
    '''
    text = ''
    if eol_code==0:
        eol_code = len(decoder_dict)-1
    ind_eol = False
    for c in text_array:
        if ind_eol==False:
            text += decoder_dict[c]
        if c==eol_code:
            ind_eol=True
    return text


def decode_response(response_array, decoder_dict):
    ''' Convert numeric codes to output sequence of text
    '''
    response_text = []
    for i in range(response_array.shape[0]):
        response_dec = [np.argmax(r) for r in response_array[i,:,:]]
        response_text += [decode_text(response_dec, decoder_dict)]
    return response_text




# Batch generator functions
#====================
def load_data_in_memory(images_list, get_target=True, encoder_dict={}, word_len=0, sample_size=1,
                       x_size=192, y_size=48, target_max_size=19, x_blanks_ini=5):
    '''Load data in memory
    Parameters:
        images_list
        get_target=True
        encoder_dict={}
        word_len: if word len=0 load all data, else load words with less or equal length
        sample_size: proportion of data selected. minimun 2 words. if 1: all data

    Returns:
        img
        img_l
        target
        target_l
    '''
    img_list = []
    img_l_list = []
    target_list = []
    target_l_list = []
    count=0
    target_len = 0
    for f in images_list:
        filename = os.path.basename(f)
        try:
            if get_target:
                target_len = len(decode_target_hex(filename.split('.')[-2].split('_')[-1]))
            
            if target_len<=word_len or word_len==0: # select only word of length less or that word_len (or all if word_len=0)
                if count<2 or random.random() <= sample_size:
                        img, target, img_l, target_l = read_word_image(f, 
                                                                       get_target=get_target, 
                                                                       encoder_dict=encoder_dict,
                                                                       x_size=x_size, 
                                                                       y_size=y_size,
                                                                       target_max_size=target_max_size,
                                                                       x_blanks_ini=x_blanks_ini)
                        img_list += [img]
                        img_l_list += [img_l]
                        target_list += [target]
                        target_l_list += [target_l]
                        count +=1
                    
        except:
            module_logger.info('Error reading the file %s',f)
        
    return np.array(img_list, dtype=np.float32),  np.array(img_l_list, dtype=np.uint8), \
           np.array(target_list, dtype=np.uint8), np.array(target_l_list, dtype=np.uint8)

    
    
    
    
def batch_generator_epoch(img, img_l, target, target_l, batch_size=32, augmentation=True):
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
        img_b      = img[batch]
        img_l_b    = img_l[batch]
        target_b   = target[batch]
        target_l_b = target_l[batch]
         
        # Augmentation
        img_b = get_img_augmented(img_b, augment=augmentation)
                   
        yield img_b, img_l_b, target_b, target_l_b
        
        
        
def batch_to_seq2seq(img_b, img_l_b, target_b, target_l_b, decoder_dict, PAD_ID, GO_ID, EOL_ID,
                     get_target=True, x_slide_size=28, slides_stride=2, seq_length=19):
    ''' Convert batch of data into seq2seq required structures
    '''
    num_classes = len(decoder_dict)
    
    weights_dict={0:10, 1:5}
    for i in range(2, 30, 1):
        weights_dict[i]=1
    
    slides_len_batch=[]
    
    batch_size = img_b.shape[0]
    decoder_inputs = np.zeros([batch_size, seq_length+1, num_classes], dtype=np.float32)
    weights = np.zeros([batch_size, seq_length+1], dtype=np.float32)
    targets = np.zeros([batch_size, seq_length+1], dtype=np.uint16)
    
    for batch_i, img in enumerate(img_b):
        
        # slides_len_batch
        max_slides = int((img.shape[1] - x_slide_size)/float(slides_stride))
        num_slides = max(2,min(max_slides, 1 + int((img_l_b[batch_i] - x_slide_size)/float(slides_stride))))
        slides_len_batch += [num_slides]

        # decoder_inputs, weigths, targets
        if get_target:
            for char_pos in range(seq_length+1):
                if char_pos == 0:
                    decoder_inputs[batch_i, char_pos, GO_ID] = 1
                    weights[batch_i, char_pos] = weights_dict[char_pos]
                    targets[batch_i, char_pos] = int(target_b[batch_i, char_pos])
                elif char_pos < target_l_b[batch_i]:
                    decoder_inputs[batch_i, char_pos, int(target_b[batch_i, char_pos-1])] = 1
                    weights[batch_i, char_pos] = weights_dict[char_pos]
                    targets[batch_i, char_pos] = int(target_b[batch_i, char_pos])
                elif char_pos == target_l_b[batch_i]:
                    decoder_inputs[batch_i, char_pos, int(target_b[batch_i, char_pos-1])] = 1
                    weights[batch_i, char_pos] = weights_dict[char_pos]
                    targets[batch_i, char_pos] = EOL_ID
                else:
                    decoder_inputs[batch_i, char_pos, PAD_ID] = 1
                    weights[batch_i, char_pos] = 0
                    targets[batch_i, char_pos] = PAD_ID
    
    return img_b, np.array(slides_len_batch), decoder_inputs, targets, weights
'''
# test
encoder_dict = create_encoder_dir()
PAD_ID = len(encoder_dict)+1
GO_ID = len(encoder_dict)+2
EOL_ID = len(encoder_dict)+3
encoder_dict[PAD_ID]='-PAD'
encoder_dict[GO_ID]='GO-'
encoder_dict[EOL_ID]='-EOL'
images_path = '/opt/data/tesis/handwriting/databases/osborne/sources/val'
img, img_l, target, target_l = load_data_in_memory(images_path, encoder_dict, word_len=3, sample_size=0.1)
gen = batch_generator_epoch(img, img_l, target, target_l)
img_b, img_l_b, target_b, target_l_b = next(gen)
img_b, slides_len_batch, decoder_inputs, targets, weights = batch_to_seq2seq(img_b, img_l_b, target_b, target_l_b,
print(slides_len_batch)
plt.imshow(img_b[4], cmap='gray')
'''






            
            
# Template function and doc            
'''
def f():
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
'''

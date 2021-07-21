# Header
env='remote'
notebook_name = 'normalize_iam'


# imports
import os
import sys
import glob
import datetime
import logging
import configparser

import argparse
import glob

import pandas as pd



import os

import math
import cv2

from subprocess import call
from sklearn import linear_model
from scipy import ndimage
import matplotlib.pyplot as plt

import numpy as np




sys.path.insert(0, '..')
import handwritting

from handwritting.read_data import encode_target_hex, decode_target_hex




#logging
know_time = datetime.datetime.now()

log_name = notebook_name+'-'+str(know_time.year)+"_"+str(know_time.month).zfill(2)+"_"+str(know_time.day).zfill(2)\
             +"_"+str(know_time.hour).zfill(2)+"_"+str(know_time.minute).zfill(2)+".log"
if not os.path.exists("logs"):
    os.makedirs("logs")


logger = logging.getLogger(notebook_name)
logging.basicConfig()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr = logging.FileHandler("logs/"+log_name)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(20)
logger.setLevel(logging.INFO)


# config
import configparser
logger.info("Load config from "+str(env)+" environment")
config = configparser.ConfigParser()
config.read('../config/'+str(env)+'.cfg')



            
            
            


iam_sources_path = '/home/ubuntu/data/tesis/handwriting/databases/IAM/sources'

#load IAM partition
# train at line level and valid and eval at page level.
partition_dict = {}

split_file_valid = os.path.join(iam_sources_path, 'split', 'valid.txt')
split_valid = set(pd.read_csv(split_file_valid, names=['c']).c)
for c in split_valid:
    partition_dict[c] = 'val'
    
split_file_eval = os.path.join(iam_sources_path, 'split', 'eval.txt')
split_eval = set(pd.read_csv(split_file_eval, names=['c']).c)
for c in split_eval:
    partition_dict[c] = 'tst'
    


    
# Read lines annotations
array_lines=[]
n=0
with open(os.path.join(iam_sources_path, 'ascii','lines.txt'), 'r') as f:
    array_words = []
    for line in f:
        if line[0] !='#':
            lp = line.strip().split(' ')
            n+=1
            array_lines.append((lp[0], lp[1], int(lp[2]), int(lp[3]), int(lp[4]), int(lp[5]), int(lp[6]), int(lp[7]), ' '.join(lp[8:])))

pd_lines = pd.DataFrame(array_lines, columns=['id_line','segmentation_result','graylevel_binarize', ' num_components',
                                              'x','y','w','h','word'])

pd_lines['id_page'] = pd_lines['id_line'].map(lambda x: '-'.join(x.split('-')[:2]))
pd_lines['id_writter'] = pd_lines['id_page'].map(lambda x: x.split('-')[0])
pd_lines['partition'] = pd_lines['id_page'].apply(lambda x: partition_dict.get(x, 'trn'))







# Read word annotations
array_words=[]
n=0
with open(os.path.join(iam_sources_path, 'ascii','words.txt'), 'r') as f:
    array_words = []
    for line in f:
        if line[0] !='#':
            lp = line.strip().split(' ')
            n+=1
            array_words.append((lp[0], lp[1], int(lp[2]), int(lp[3]), int(lp[4]), int(lp[5]), int(lp[6]), lp[7], ' '.join(lp[8:])))

pd_words = pd.DataFrame(
    array_words,
    columns=[
        'id_word','segmentation_result','graylevel_binarize',
        'x','y','w','h','grammar_tag','word'
    ]
)

pd_words['line'] = pd_words.id_word.apply(lambda x: '-'.join(x.split('-')[:-1]))
pd_words['page'] = pd_words.line.apply(lambda x: '-'.join(x.split('-')[:-1]))

# marca de las palablas a seleccionar
pd_words['selected'] = False
pd_words.loc[(pd_words.segmentation_result == 'ok') & (pd_words.word != "#") & (pd_words.x > 0), 'selected'] = True

lines_selected = set(pd_words[pd_words.selected].line.values)





def enhace_text_image(img_file, out_file, imgtxtenh_path='/opt/imgtxtenh/src/imgtxtenh'):
    ''' Enhace text using imgtxtenh
    '''
    return_call = call([imgtxtenh_path, "-d", "118.110", "-V", "-", img_file, out_file])
    return return_call




def slant_angle(img, treshold_up=100, treshold_down=100):
    """ Calculate slant angle. Cursive
        - Check the upper neighboords of pixels with left blank
        - Utilizar despeus de hacer una mejora de contraste.
        - Usar despues de habr corregido el slope de la linea
    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255 
        treshold_up: umbral de gris para decidir que algo es negro
        treshold_down: umbral de gris pra decidir que algo es blanco
    """
    angle = []
    # Contadores para pixeles Centrales, Left y Right
    C = 0
    L = 0
    R = 0
    for w in range(1,img.shape[1]-1):
        for h in range(2,img.shape[0]-1):
            if img[h,w] > treshold_up and img[h, w-1] < treshold_down: # si pixel negro y blanco a la izquierda..
                if img[h-1, w-1] > treshold_up: # si arriba izquierda es negro
                    L +=1
                    angle += [-45*1.25]
                elif img[h-1, w] > treshold_up: # si arriba centro es negro
                    C += 1
                    angle += [0]
                elif img[h-1, w+1] > treshold_up: # si arriba derecha es negro
                    R += 1
                    angle += [45*1.25]
    logger.debug(f"Slant angle. Left, Center, Rigth, Angle: {L}, {C}, {R}, {(R-L)/(L+C+R)}")
    return np.arctan2((R-L),(L+C+R))



def correct_slant(img, treshold=100):
    """Corrige slant del texto. Cursiva
    
    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255 
        treshold: 
    """
    # Estimate slant angle
    angle = slant_angle(img, treshold_up = treshold, treshold_down = treshold)
    
    # convert image to to negative
    img = 255 - img
    
    # Add blanks in laterals to compensate the shear transformation cut 
    if angle>0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0]*angle)])], axis=1)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0]*(-angle))]), img], axis=1)
        
    # Numero de columnas añadidas a la imagen
    # positions//2 permiten ajusta las posiciones de cada palabra si se tiene segmentandas antes de esta transformación
    positions = int(abs(img.shape[0]*angle))
        
    # shear matrix and affine transformation
    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    img2 = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    )
    
    
    return img2, angle, positions//2



def get_x_positions_line(pd_words, line_img, line, x_line, inc_positions):
    """ recupera las tupas de posiciones x de cada palabra de una linea de IAM
    y las corrige una posición x_line
    """
    x_positions = []
    words_list = []
    id_words_list = []
    for t in pd_words[(pd_words.line == line) & (pd_words.selected)].itertuples():
        x_positions += [(t.x - x_line + inc_positions, t.x - x_line + inc_positions + t.w)]
        words_list += [t.word]
        id_words_list += [t.id_word]
    logger.debug(f"x_positions: {x_positions}")
        
        
    x_positions_correct = []
    x_prev = 0
    for i in range(len(x_positions)-1):
        new_x = (x_positions[i][1] + x_positions[i+1][0]) // 2
        x_positions_correct += [(x_prev, new_x)]
        x_prev = new_x
    x_positions_correct += [(x_prev, x_positions[-1][1])] 
    
    
    x_positions_correct[-1] = (x_positions_correct[-1][0], line_img.shape[1])
    
    logger.debug(f"x_positions_correct: {x_positions_correct}")
    
    
    img_list = []
    for (x1,x2) in x_positions_correct:
        if x2-x1>=2:
            img_list += [line_img[:, x1:x2]]
        else:
            img_list += [line_img[:, x1:x1+2]]
            
    return img_list, id_words_list, words_list
    

    
    
    
def find_outliers(img, pct_th=0.75):
    '''find_outliers
    '''
    # vertica histogram
    h_sum = np.sum(img, axis=1)
    #Detect outliers in histogram
    pct_threshold = (np.max(h_sum) - np.min(h_sum))*pct_th
    h_outliers = [p[0] for p in np.argwhere(h_sum > pct_threshold)] + [img.shape[0]]

    # vertica histogram
    v_sum = np.sum(img, axis=0)
    #Detect outliers in histogram
    pct_threshold = (np.max(v_sum) - np.min(v_sum))*pct_th
    v_outliers = [p[0] for p in np.argwhere(v_sum > pct_threshold)] + [img.shape[1]]
    
    return h_outliers, v_outliers




def detect_baseline(img, treshold=20):
    '''
    detect baseline
    '''
    
    low = []
    for w in range(1,img.shape[1]-1):
        if np.max(img[:,w]) > treshold:
            for h in range(img.shape[0]-5, 0, -1):
                if img[h,w] > treshold:
                    low += [[h,w]]
                    break
    points_lower = np.array(low)
    
    #Robust outliers regression
    x = points_lower[:,1].reshape(points_lower.shape[0],1)
    y = points_lower[:,0].reshape(points_lower.shape[0],1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(x, y)
    y0 = model_ransac.predict(x[0].reshape(1, -1))[0][0]
    y1 = model_ransac.predict(x[-1].reshape(1, -1))[0][0]
    y_mean = model_ransac.predict(np.array([img.shape[1]/2]).reshape(1, -1))
    angle = np.arctan((y1 - y0) / (x[-1] - x[0])) * (180 / math.pi)
    
    return y0, y1, int(y_mean), angle[0]




def detect_upperline(img, treshold=20):
    '''
    detect baseline
    '''
    upp = []
    for w in range(1,img.shape[1]-1):
        if np.max(img[:,w]) > treshold:
            for h in range(5, img.shape[0]):
                if img[h,w] > treshold:
                    upp += [[h,w]]
                    break
    points_upper = np.array(upp)
    
    #Robust outliers regression
    x = points_upper[:,1].reshape(points_upper.shape[0],1)
    y = points_upper[:,0].reshape(points_upper.shape[0],1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(x, y)
    y0 = model_ransac.predict(x[0].reshape(1, -1))[0][0]
    y1 = model_ransac.predict(x[-1].reshape(1, -1))[0][0]  
    y_mean = int(model_ransac.predict(np.array([img.shape[1]/2]).reshape(1, -1)))
    
    return y0, y1, y_mean





def crop_borders(img):
    ''' Crop borders
    '''
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(img)
    if len(true_points)>0:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        img = img[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                  top_left[1]:bottom_right[1]+1]  # inclusive
    return img



def rescale(img, threshold=20):
    img[img < 0] = 0
    img2 = np.array((img - np.min(img)) * (255 / (np.max(img)-np.min(img)) ) )
    img2[img2 < threshold] = 0
    return img2



def correct_line_inclination(img):

    #Detect baseline to correct inclination
    
    y0, y1, y_mean, angle = detect_baseline(img)
    logger.debug(f"rotate angle: {angle}")
    
    # Correct inclination with lower angle
    img_out = ndimage.rotate(img, angle)

    img_out = rescale(img_out)
    
    img_out = crop_borders(img_out)
    
    return img_out

    
    
def reshape_areas(img, treshold=20):

    #Detect baseline
    y0_base, y1_base, y_mean_base, angle = detect_baseline(img, treshold=treshold)
    logger.debug(f'Baseline:, {y0_base}, {y1_base}, {angle}')

    #Estimate Upper line
    y0_upper, y1_upper, y_mean_upper = detect_upperline(img, treshold=treshold)
    logger.debug(f'Upperline:, {y0_upper}, {y1_upper}')

    #Posiciones y de cada franja
    position_upper = max(0, int(min(y0_upper, y1_upper)))
    position_base = min(img.shape[0], int(max(y0_base, y1_base)))
    
    # Altura de cada franja
    h_upper = position_upper
    h_base = img.shape[0] - position_base
    h_core = position_base - position_upper
    
    #correccion de posiciones si core es pequeño
    if h_upper>5 & h_upper > h_core:
        position_upper -= 5
    if h_base > 5 & h_base > h_core:
        position_base += 5
    
    
    # Altura de cada franja
    h_upper = position_upper
    h_base = img.shape[0] - position_base
    h_core = position_base - position_upper
    logger.debug(position_upper, position_base, '-', h_upper, h_core, h_base)
    
    
    # Ajuste de la parte de descenders
    if h_base > h_core:
        r_baseline = max(h_core/h_base, 0.5)
        img_inf_rescaled = cv2.resize(img[position_base:,:], (img.shape[1], int(h_base * r_baseline)))
    elif h_base < 20:
        img_inf_rescaled = np.concatenate((img[position_base:,:], np.zeros((20-h_base, img.shape[1]))), axis=0)
    else:
        img_inf_rescaled = img[position_base:,:]
    
    # Ajuste de la parte de ascenders
    if h_upper > h_core:
        r_upperline = max(h_core/h_upper, 0.5)
        img_sup_rescaled = cv2.resize(img[:position_upper,:], (img.shape[1], int(h_upper * r_upperline)))
    elif h_upper < 20:
        img_sup_rescaled = np.concatenate((np.zeros((20-h_upper, img.shape[1])), img[:position_upper,:]), axis=0)
    else:
        img_sup_rescaled = img[:position_upper,:]
    
        
    logger.debug(f"Rescale areas: {h_upper}, {h_core}, {h_base} {img_inf_rescaled.shape} {img_sup_rescaled.shape}")
    
    
    img_rescaled = np.concatenate((img_sup_rescaled, img[position_upper:position_base,:], img_inf_rescaled), axis=0)


    return img_rescaled
    
    
    
    
# Normalize size of each word
def normalize_shape(img_list, x_size=192, y_size=48, plot=False):
    
    img_normalized_list = []

    for img in img_list:
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

        img_normalized_list += [img]
    
    if plot:
        fig = plt.figure()
        n = 1
        for img, img_norm in zip(img_list, img_normalized_list): 
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Original')
            fig.tight_layout()
            plt.imshow(255-img, cmap='gray')
            n += 1
            
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Normalized')
            fig.tight_layout()
            plt.imshow(255-img_norm, cmap='gray')
            n += 1
        
    return img_normalized_list



def proccess_line(t, pd_words, iam_sources_path, destination_folder, plot=False):
        #logger.info(t)

        # Read the entire page
        page_filename = os.path.join(
            iam_sources_path,
            'forms',
            t.id_page+'.png'
        )


        # enhace contrast in entire page
        return_call = enhace_text_image(
            page_filename,
            '/tmp/imgenh_trn.png',
            imgtxtenh_path='/home/jorge/soft/imgtxtenh/src/imgtxtenh'
        )
        page_img_enhaced = cv2.imread('/tmp/imgenh_trn.png', cv2.IMREAD_GRAYSCALE)


        # crop line from page
        line_img = page_img_enhaced[t.y:t.y+t.h, t.x:t.x+t.w]


        # Correct slope in line
        try:
            horizontal_line_img = correct_line_inclination(255-line_img)
        except:
            horizontal_line_img = 255-line_img
            logger.info(f'ERROR en correct slope. Line: {t.id_line}')
            logger.info(f"Error: {sys.exc_info()[0]}")
                    
        # Correct slant in line
        try:
            line_img_no_slant, angle, inc_positions = correct_slant(255-horizontal_line_img)
            logger.debug(f"Slant primera correccion: {angle} | {inc_positions}")
            # segundo pase si angulo mayor que > 3
            if abs(slant_angle(line_img_no_slant)) > 0.02:
                line_img_no_slant, angle_2, inc_positions_2 = correct_slant(255-line_img_no_slant)
                inc_positions = inc_positions + inc_positions_2
                logger.debug(f"Slant segunda correccion: {angle_2} | {inc_positions_2}")
        except:
            line_img_no_slant = horizontal_line_img
            inc_positions = 0
            logger.info(f'ERROR en correct slant. Line: {t.id_line}')
            logger.info(f"Error: {sys.exc_info()[0]}")
                    

            
        # Redimensionamiento de areas de ascenders y descenders
        try:
            img_rescaled = reshape_areas(line_img_no_slant, treshold=20)
        except:
            img_rescaled = line_img_no_slant
            logger.info(f'ERROR en redimensionamiento. Line: {t.id_line}')
            logger.info(f"Error: {sys.exc_info()[0]}")
                    

            
        plt.rcParams['figure.figsize'] = (20, 10)     
        if plot:    
            plt.figure()
            plt.imshow(line_img, cmap='gray')

            plt.figure()
            plt.imshow(255-horizontal_line_img, cmap='gray')

            plt.figure()
            plt.imshow(255-line_img_no_slant, cmap='gray')

            plt.figure()
            plt.imshow(255-img_rescaled, cmap='gray')


        #separe in words
        img_words_list, id_words_list, words_list = get_x_positions_line(pd_words, img_rescaled, t.id_line, t.x, inc_positions)

        # Normalize shape and size of the words
        plt.rcParams['figure.figsize'] = (20, 50)     
        words_normalized_list = normalize_shape(img_words_list, x_size=192, y_size=48, plot=plot)

        # Save words
        for im_gray, name, word in zip(words_normalized_list, id_words_list, words_list):
            filename = os.path.join(destination_folder, name + '_' + encode_target_hex(word) + '.png')
            logger.debug(filename)
            cv2.imwrite(filename, im_gray)
            


destination_folder = '/home/ubuntu/data/tesis/handwriting/databases/IAM/normalized_test/trn'
if not(os.path.exists(destination_folder)):
    os.makedirs(destination_folder)    

for t in pd_lines[(pd_lines.id_line.isin(lines_selected)) & (pd_lines.partition == 'trn')].itertuples():
    try:
        proccess_line(t, pd_words, iam_sources_path, destination_folder)
    except:
        logger.info(f"LINE ERROR: {t}")
        logger.info(f"Error: {sys.exc_info()[0]}")
                

    
    

# Functions to normalize handwritting images



import os
import glob
import logging
import random 

import numpy as np

from subprocess import call
from PIL import Image

import cv2 # input first numpy


# logging
logger = logging.getLogger(__name__)


# Image augmentation functions
# ========================
def move_img(img):
    '''Move image horizontaly a [0,10] random pixels'''
    pixels_move = 1 + int(random.random()*10)
    img2 = np.ones_like(img)*0
    img2[:,pixels_move:] = img[:,:-pixels_move]
    return img2

def resize_down(img):
    '''Resize down an image a randos factor'''
    factor = 0.95 - random.random()/4.
    h_ini, _ = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img)*0
    img2[(h_ini-h_fin)//2:-(h_ini-h_fin)//2, :w_fin] = img1
    return img2

def resize_up(img):
    '''Resize up an image a randon factor'''
    factor = 1 + random.random()/8.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, _ = img1.shape
    img2 = img1[h_fin-h_ini:, :w_ini]
    return img2


#PENDING
def add_noise_to_image(img):
    '''Add snow noise to an image'''
    return img



def get_img_augmented(image_list, augment=True):
    ''' Generate augmented images for a list of images

    '''
    if augment: 
        augmented_image_list = []
        for img in image_list:
            if len(img.shape)>2:
                img = img[:,:,0]

            # Move left
            img = move_img(img)

            # Skew
            if random.random() < 0.8 :
                angle = (random.random()-0.5)/2.2
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






# Image normalization functions
# =============================


def shear_angle(img, treshold_up=100, treshold_down=100):
    '''Find shear angle:
    - Check the upper neighboords of pixels with left blank
    '''
    angle = []
    C = 0
    L = 0
    R = 0
    for w in range(1,img.shape[1]-1):
        for h in range(2,img.shape[0]-1):
            if img[h,w] > treshold_up and img[h, w-1] < treshold_down:
                if img[h-1, w-1] > treshold_up:
                    L +=1
                    angle += [-45*1.25]
                elif img[h-1, w] > treshold_up:
                    C += 1
                    angle += [0]
                elif img[h-1, w+1] > treshold_up:
                    R += 1
                    angle += [45*1.25]
    return np.arctan2((R-L),(L+C+R))


def shear_angle2(img, treshold_up=100, treshold_down=100):
    '''Find shear angle V2:
    - Check the upper neighboords of pixels with left blank
    - Need a positive image
    '''
    C = 0
    L = 0
    R = 0
    for w in range(1,img.shape[1]-1):
        for h in range(2,img.shape[0]-1):
            if img[h,w] > treshold_up and ( img[h, w+1] < treshold_down):
                if img[h-1, w-1] > treshold_up and img[h+1, w-1]  < treshold_down:
                    if img[h+1, w] > treshold_up:
                        L += 1
                        C += 1
                    if img[h+1, w+1] > treshold_up:
                        L += 2
                elif img[h-1, w] > treshold_up:
                    if img[h+1, w-1] > treshold_up:
                        R += 1
                        C += 1
                    if img[h+1, w] > treshold_up:
                        C += 2
                    if img[h+1, w+1] > treshold_up:
                        L += 1
                        C += 1
                elif img[h-1, w+1] > treshold_up:
                    if img[h+1, w-1] > treshold_up:
                        R += 2
                    if img[h+1, w] > treshold_up:
                        C += 1
                        R += 1
    return np.arctan2((R-L),(L+C+R))



def correct_shear(img, treshold=100):
    ''' corrrect shear
    '''
    # Estimate angle
    angle = shear_angle(img, treshold_up = treshold, treshold_down = treshold)
    
    # convert to negative
    img = 255- img
    
    # add blanks at the rigth to compensate the shear transformation cut 
    if angle>0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0]*angle)])], axis=1)
        positions = int(img.shape[0]*angle)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0]*(-angle))]), img], axis=1)
        positions = int(img.shape[0]*(-angle))
        
    # shear matrix and affine transformation
    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    img2 = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    )
    
    
    return img2, angle, positions//2


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


#add borders
def add_borders(img, border_size_x=5, border_size_y=5):
    '''Add borders to the image
    '''
    if len(img.shape)>2:
        img = img[:,:,0]
    y, x = img.shape
    img = np.concatenate([np.zeros([y, border_size_x]), img, np.zeros([y, border_size_x])], axis=1)
    y, x = img.shape
    img = np.concatenate([np.zeros([border_size_y, x]), img, np.zeros([border_size_y, x])], axis=0)
    return img



def enhace_text_image(img_file, out_file, imgtxtenh_path='/opt/imgtxtenh/src/imgtxtenh'):
    ''' Enhace text using 
    '''
    return_call = call([imgtxtenh_path, "-d", "118.110", img_file, out_file])
    return return_call



def fit_baseline(img):
    '''Fit the baseline  
    '''
    
    # percen of pixels by row
    rows = np.sum(np.clip(img, 0, 1), axis=1)/img.size[1]
    
    # upper and lower positions finded from the max position
    max_pos = np.argwhere(rows > np.max(rows)*0.9)
    
    for y_upp in range(max_pos[0][0], 0, -1):
        if rows[y_upp] < np.max(rows)/4:
            break

    for y_low in range(max_pos[-1][0], len(rows)):
        if rows[y_low] < np.max(rows)/4:
            break
    
    # Sizes old and new
    size_upp = y_upp
    size_cen = y_low - y_upp
    size_low = img.size[1] - y_low
    new_size_upp = int(max(size_cen*0.6, size_upp*0.6))
    new_size_low = int(max(size_cen*0.6, size_low*0.6))
    
    #resize by parts
    im_upp  = img.crop((0, 0, img.size[0], y_upp))
    im_upp2 = im_upp.resize((img.size[0], new_size_upp))
    im_cen  = img.crop((0, y_upp, img.size[0], y_low))
    im_low  = img.crop((0, y_low, img.size[0], img.size[1]))
    im_low2 = im_low.resize((img.size[0], new_size_low))
    im_concat = np.concatenate([im_upp2, im_cen, im_low2], axis=0)
    
    return Image.fromarray(im_concat)


def adjust_img_x(img, th_low=20, th_upp=150, factor_base=0.1):
    '''histogram to identify number of changes
    Don't work
    '''
    img2 =np.asarray(img)
    num_changes = []
    for i in range(img2.shape[0]):
        num_changes_row = 0
        level = 0
        for v in img2[i,:]:
            if level==0 and v>th_upp:
                num_changes_row+=1
                level=1
            if level==1 and v<th_low:
                num_changes_row+=1
                level=0
        num_changes += [num_changes_row/img2.shape[1]]
    factor_scale = factor_base/np.mean(np.sort(num_changes)[-3:])
    if img.size[0]*1.5<img.size[1]: # if long word factor between [0.7, 1]
        factor_scale = min(max(factor_scale, 1), 0.7)
    else: # if sort word factor between [1, 1.3]
        factor_scale = min(max(factor_scale, 1.3), 1)
    new_size_x = int(img.size[0] * factor_scale)

    img3 = img.resize((new_size_x, img.size[1]))
    return num_changes, img3
                    


def normalize_word(filename, destination_path,
                   umbralize_threshold=5,
                   ysize=None,
                   min_y_size=0,
                   temp_dir='/tmp',
                   border_size_x=5,
                   border_size_y=5,
                   adjust_baseline=False,
                   imgtxtenh_path='/opt/imgtxtenh/src/imgtxtenh',
                   shear_correction=True
                  ):
    '''
    Steps:
        - Enhace text image file
        - Read and convert to gray
        - Correct sheat and invert
        - Umbralize low vaues to zeros
        - Crop zero borders
        - Add standard borders
        - Resize to fix xsize
        - Save
    '''
    
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    name = os.path.basename(filename)
    destination_name = os.path.join(destination_path, '.'.join(name.split('.')[:-1])+'.png')
    temp_filename = os.path.join(temp_dir,"imgtxtenh_out.png")
    
    return_call = enhace_text_image(filename, temp_filename, imgtxtenh_path=imgtxtenh_path)
    
    #return_call = call(["convert", "/tmp/imgtxtenh_out.png", "-deskew", "40%",
    #                    "+repage", "-strip",
    #                    os.path.join(temp_dir,"convert_out.png")])
    angle = 0        
    if return_call==0:
        image = cv2.imread(temp_filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if shear_correction:
            image2, angle, _ = correct_shear(gray_image)
        else:
            image = 255 - image
            M = np.float32([[1, 0, 0], [0, 1, 0]])
            image2 = cv2.warpAffine(image,M,(image.shape[1], image.shape[0]),
                          flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
            angle = 0
        
        if image2.shape[0]>1 and image2.shape[1]>1:
            # Umbralize
            image2 = (image2 > umbralize_threshold) * image2

            #Crop borders
            image3 =crop_borders(image2)

            #Add standard border
            image4 = add_borders(image3, border_size_x=5, border_size_y=5)

            # resize to fixed y
            y, x  =image4.shape
            if ysize is not(None):
                image4 = cv2.resize(image4, (int(x*(ysize/y)), ysize))

            # if the image are y small, add borders to up and down until minimun size 
            if y < min_y_size:
                image4 = add_borders(image4, border_size_x=0, border_size_y=int((min_y_size-y)/2)+1)
                
            # adjust size
            if adjust_baseline:
                image4 = fit_baseline(Image.fromarray(image4))
                image4 = np.asarray(image4)

            # save
            cv2.imwrite(destination_name, image4)
        else:
            logger.info('Error: Image %s with dimensions %s', filename, str(image2.shape))
            return None, None, None

    else:
        logger.info('Error processing %s image with imgtxtenh', filename)
        return None, None, None
            
    return image4, destination_name, angle
    

    
    
def normalize_dir(input_path,
                  destination_path, 
                  extensions_list=['tiff', 'jpg', 'png', 'gif'],
                  min_y_size=0,
                  adjust_baseline=False):
    ''' Normalize all images in the input_path
    '''
    
    if not(os.path.exists(destination_path)):
        os.mkdir(destination_path)
        
    source_list = []
    angle_list = []
    destination_list = []
    
    for extension in extensions_list:
        list_dir = glob.glob(os.path.join(input_path, "*."+extension))

        for f in list_dir:
            _, destination_name, angle = normalize_word(f, destination_path, 
                                                     min_y_size=min_y_size,
                                                     adjust_baseline=adjust_baseline)
            source_list += [f]
            angle_list += [angle]
            destination_list += [destination_name]
        
    return source_list, angle_list, destination_list
    
    
 

def resize_img(img, ysize=42, max_xsize=150):
    '''Resize image to fixed size por input model
    img
    ysize=42
    max_xsize=150
    '''
    # if is array, convert to pillow. grayscale
    if type(img) == np.ndarray:
        img = Image.fromarray(np.asarray(img, dtype=np.uint8), 'L')


    wpercent = (ysize/float(img.size[1]))
    xsize = int((float(img.size[0])*float(wpercent)))
    img = img.resize((xsize, ysize), Image.ANTIALIAS)
    
    if xsize < max_xsize:
        img_final = np.zeros((ysize, max_xsize))
        img_final[:,:xsize] = np.asarray(img)
    else:
        img_final = img.resize((max_xsize, ysize), Image.ANTIALIAS)
        img_final = np.asarray(img_final)
    return img_final


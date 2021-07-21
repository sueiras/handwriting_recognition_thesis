
# Image augmentation code
# ========================

import itertools
import random
import warnings
from math import floor

import cv2
import numpy as np
from PIL import Image
from skimage import transform as stf


def ElasticDistortion(img_array, max_kernel = 3, max_magnitude = 20, min_h_sep=1, min_v_sep=1):
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    
    magnitude = np.random.randint(1, max_magnitude + 1)
    magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
    kernel = np.random.randint(1, max_kernel + 1)
    horizontal_tiles = kernel
    vertical_tiles = kernel
    
    img = Image.fromarray(img_array*255)
        
    w, h = img.size

    width_of_square = int(floor(w / float(horizontal_tiles)))
    height_of_square = int(floor(h / float(vertical_tiles)))

    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

    dimensions = []
    shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])

            sm_h = min(magnitude_w, width_of_square - (min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                0])) if horizontal_tile > 0 else magnitude_w
            sm_v = min(magnitude_h, height_of_square - (min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                1])) if vertical_tile > 0 else magnitude_h

            dx = random.randint(-sm_h, magnitude_w)
            dy = random.randint(-sm_v, magnitude_h)
            shift[vertical_tile][horizontal_tile] = (dx, dy)

    shift = list(itertools.chain.from_iterable(shift))

    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for id, (a, b, c, d) in enumerate(polygon_indices):
        dx = shift[id][0]
        dy = shift[id][1]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1,
                        x2, y2,
                        x3 + dx, y3 + dy,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1,
                        x2 + dx, y2 + dy,
                        x3, y3,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1,
                        x2, y2,
                        x3, y3,
                        x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy,
                        x2, y2,
                        x3, y3,
                        x4, y4]

    generated_mesh = []
    for i in range(len(dimensions)):
        generated_mesh.append([dimensions[i], polygons[i]])

    generated_mesh = generated_mesh

    return np.array(img.transform(img.size, Image.MESH, generated_mesh, resample=Image.BICUBIC))/255.




def RandomTransform(img_array, max_val=16):
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    img = Image.fromarray(255 - img_array*255)
    w, h = img.size

    dw, dh = (max_val, 0) if random.randint(0, 2) == 0 else (0, max_val)

    def rd(d):
        return random.uniform(-d, d)

    def fd(d):
        return random.uniform(-dw, d)

    # generate a random projective transform
    # adapted from https://navoshta.com/traffic-signs-classification/
    tl_top = rd(dh)
    tl_left = fd(dw)
    bl_bottom = rd(dh)
    bl_left = fd(dw)
    tr_top = rd(dh)
    tr_right = fd(min(w * 3 / 4 - tl_left, dw))
    br_bottom = rd(dh)
    br_right = fd(min(w * 3 / 4 - bl_left, dw))

    tform = stf.ProjectiveTransform()
    tform.estimate(np.array((
        (tl_left, tl_top),
        (bl_left, h - bl_bottom),
        (w - br_right, h - br_bottom),
        (w - tr_right, tr_top)
    )), np.array((
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    )))

    # determine shape of output image, to preserve size
    # trick take from the implementation of skimage.transform.rotate
    corners = np.array([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ])

    corners = tform.inverse(corners)
    minc = corners[:, 0].min()
    minr = corners[:, 1].min()
    maxc = corners[:, 0].max()
    maxr = corners[:, 1].max()
    out_rows = maxr - minr + 1
    out_cols = maxc - minc + 1
    output_shape = np.around((out_rows, out_cols))

    # fit output image in new shape
    translation = (minc, minr)
    tform4 = stf.SimilarityTransform(translation=translation)
    tform = tform4 + tform
    # normalize
    tform.params /= tform.params[2, 2]

    img = stf.warp(np.array(img), tform, output_shape=output_shape, cval=255, preserve_range=True)
    img = stf.resize(img, (h, w), preserve_range=True).astype(np.uint8)

    return (255 - img) / 255



def move_img(img, max_move=10):
    pixels_move = 1 + int(random.random()*max_move)
    img2 = np.ones_like(img)*0
    img2[:, pixels_move:] = img[:, :-pixels_move]
    return img2

def resize_down(img, max_factor=0.1):
    factor = 0.95 - random.random() * max_factor
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img)*0
    img2[(h_ini-h_fin)//2:-(h_ini-h_fin)//2, :w_fin] = img1
    return img2

def resize_up(img, max_factor=0.1):
    factor = 1 + random.random() * max_factor
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = img1[h_fin-h_ini:, :w_ini]
    return img2






def img_augmented(img):


    if random.random() < 0.5:
        img = ElasticDistortion(img)

    if random.random() < 0.5:
        img = RandomTransform(img)

    # Move left
    #img = move_img(img)

    # Skew
    if random.random() < 0.8 :
        shape_ini = img.shape
        angle = (random.random()-0.5)/3.
        M = np.float32([[1, -angle, 0.5*img.shape[0]*angle], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

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
    
    img = img/np.max(img)

    return img


import drawBot
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_text(text, font_file,
              tracking=-5,
              y_factor=1.5,
              font_size=180,
              save_file="/tmp/text.png"):
    ''' Draw a text and save it
        - tracking: interletter space []
    '''
    x_shape = (font_size/2.5) * len(text)*2
    #x_shape = ((font_size/4)+tracking) * len(text)*2
    y_shape = font_size*2
    
    drawBot.newDrawing()
    drawBot.newPage(x_shape, y_shape)
    drawBot.tracking(tracking)
    drawBot.font(font_file)
    drawBot.fontSize(font_size)
    #drawBot.openTypeFeatures(clig=True)
    #print(drawBot.listOpenTypeFeatures())
    drawBot.text(text, (5, 5 + font_size/1.5))
    drawBot.saveImage(save_file)
    drawBot.endDrawing()

    #Read and keep the last chanel and convert to array
    img = plt.imread(save_file)[:,:,-1]
    img = (1 - img)*255 
    plt.imsave(save_file, img, cmap='gray')

    return img


def generate_font_batch(text_list, augment=True, save_file="/tmp/text.png"):
    '''generate_font_batch
    '''
    imgs = []
    imgs_len = []
    for text in text_list:
        tracking  =np.random.randint(-10, 0)
        img_len = draw_text(text, tracking=tracking, save_file=save_file)
        imgs += [plt.imread(save_file)[:,:,-1]]
        imgs_len += [img_len]
        print(plt.imread(save_file)[:,:,-1].shape)
   
    #imgs = get_img_augmented(imgs, augment=augment)
    
    return np.array(imgs, dtype=np.float32), np.array(imgs_len, dtype=np.uint8)




def change_shear_generated_text(img, angle):
    ''' Apply a shear to the text
    Need a positive image
    Return a positive image
    '''
    # Convert to negative
    img = 255 - img
    
    # add blanks at the rigth to compensate the shear transformation cut
    if angle>0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0]*angle)])], axis=1)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0]*(angle*(-1)))]), img], axis=1) 
        
    # shear matrix and transformation
    M = np.float32([[1, angle*(-1), 0], [0, 1, 0]])
    img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    
    # convert to positive
    img2 = 255 - img2
 
    return img2
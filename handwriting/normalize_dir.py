# Header
env='remote'
notebook_name = 'normalize_dir'


# imports
import os
import sys
import glob
import datetime
import logging
import configparser

import argparse
import glob

sys.path.insert(0, '..')
import handwritting

from handwritting.normalization import normalize_word


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
logger.setLevel(logging.DEBUG)


# config
import configparser
logger.info("Load config from "+str(env)+" environment")
config = configparser.ConfigParser()
config.read('../config/'+str(env)+'.cfg')


# args
args = argparse.ArgumentParser()
args.add_argument("--input_dir", type=str, default='/home/jorge/data/handwritting/fonts_generated/fonts_rimes_02/original/trn', help="input_dir")
args.add_argument("--output_dir", type=str, default='/home/jorge/data/handwritting/fonts_generated/fonts_rimes_02/normalized/trn', help="output_dir")
args.add_argument("--ysize", type=int, default=48, help="ysize")


#TODO
#Pending add a labels file to include it hex encoded in the end of the name if is provided. If not use the same name as the original image 



FLAGS, unparsed = args.parse_known_args()
print("\nParameters:", FLAGS)


#  Normalization
image_filename_list = glob.glob(os.path.join(FLAGS.input_dir, '*.png'))

for i,image_filename in enumerate(image_filename_list):
    if i%1000==0:
        print(i)
    try:
        img, img_file, angle = normalize_word(image_filename,
                                              FLAGS.output_dir,
                                              imgtxtenh_path=config['LOCAL_PATHS']['imgtxtenh_path'],
                                              shear_correction=True,
                                              ysize=FLAGS.ysize)
    except:
        print('Error converting', image_filename)

        

        

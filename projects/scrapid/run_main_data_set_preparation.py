###Import
import glob
import timeit
import numpy as np
import cv2 as cv
import os

import matplotlib.pyplot as plt
import matplotlib
import copy
from matplotlib import colors
from math import pi
import os
import re

#To run project notebooks in subfolders as from root folder
import sys
sys.path.append('../../')

###Import self-written library
import featexlib
from featexlib import Debug as dbug
from featexlib import *

###Active cell###
#Y is in RGB format or one channel
sp_y_is_rgb = False
sp_cv_read_mode = cv.IMREAD_COLOR if sp_y_is_rgb else cv.IMREAD_UNCHANGED
interp_y = cv.INTER_NEAREST #Remove interpolation for y class

###ACTIVE CELL###
###Get names with of checked images from folder.
filtered_list = Image_generator.get_file_names_in_folder(files_dir ="../../Data/scrapid/images_and_annotations/1_channel/checkedannotations")
print('Found file names:', len(filtered_list))

###ACTIVE CELL###
###FROM ONE CHANNEL Y
###Prepare X dataset, reading from directory

images_x, images_y_raw = Image_generator.load_from_folders_xy(
    save_dir_x = "../../Data/scrapid/images_and_annotations/1_channel/images_protocols_matched",
    save_dir_y = "../../Data/scrapid/images_and_annotations/1_channel/masks",
    name_list = filtered_list,
    size=(3557,1071),
    file_filter_regex = r'/([^/]+)\.png',
    cv_read_mode = sp_cv_read_mode,
    recursive = True,
    interp_y = interp_y
)
print('Num classes', images_y_raw.max()+1)
print('Shape',images_y_raw.shape)
print('Unique classes:', np.unique(images_y_raw))

###ACTIVE CELL###
#Configure image generation pipeline, based on Image_generator class
bill_img_gen = Image_generator(
    Image_generator.Pipeline_x_y_images(
        common_layers = [
            Image_generator.Mod_reshape(shape=(...,None),target="y"),
            Image_generator.Mod_resize(size=(768,256), target="x"),
            Image_generator.Mod_resize(size=(768,256), target="y", interpolation = cv.INTER_NEAREST),
            Image_generator.Mod_split(size=(256,256), target="all"),
            Image_generator.Mod_duplicate(num=2)
            ###interpolation cv.INTER_NEAREST is required!!!!!
        ],
        special_layers = [
                [
                    Image_generator.Mod_shift(sh_h=100, sh_v=100, rand = True, interpolation_saved = cv.INTER_NEAREST),
                    Image_generator.Mod_rotate(angle=180, rand = True, interpolation_saved = cv.INTER_NEAREST)
                ]
        ]
    )
)

#Generate images. Run twice while need to export images_y_raw color images. 
x_result, images_y_raw = bill_img_gen(images_x, images_y_raw)

del images_x

#Show some result images
print('x.shape =',x_result.shape,',', 'y.shape =',images_y_raw.shape)

###ACTIVE CELL###
###Make multi class y array

#List of codes for RGB y, do not change the order!
list_of_codes_colors = [
    0, #Black
    64, #Dark red
    32896, #Yellow
    8388608, #Blue
    32768, #Green
    192, #Red
    8388736, #Violet
    4194448, ##????
    20624, ##????
    52443, ##????
    10911041, ##????
]

##For RGB y
list_of_codes = list_of_codes_colors

###FOR ONE CHANNEL Y
if not sp_y_is_rgb:
    list_of_codes = [i for i in range(images_y_raw.max()+1)]
    
#Make 1-hot coded y-multiarray from multiclass picture
def get_classes_from_multiclass(inp_y, rgb = True):
    y_colors = np.array([])
    if rgb:
        y_colors = (inp_y[:,:,:,0]*65536+inp_y[:,:,:,1]*256+inp_y[:,:,:,2])[...,None] 
    else:
        y_colors = inp_y[...,None]
    out = []
    for c in list_of_codes:
        out.append((y_colors==c).astype(int)) 
    return np.concatenate(out, axis=3)

y_result = get_classes_from_multiclass(images_y_raw, rgb=sp_y_is_rgb)

print('y.shape',y_result.shape)


'''
###ACITVE CELL#####
###Making RBG y pics from one channel data
#Help function to get RGB codes from DEC


def get_rgb_from_codes(inp):
    return np.array([[i//65536,(i - 65536*(i//65536))//256, i - 65536*(i//65536) - 256*((i - 65536*(i//65536))//256)] for i in inp]).astype(np.uint8)

def multichannel_to_rgb(inp):

    list_of_colors = get_rgb_from_codes(list_of_codes_colors)
    
    res_r = np.argmax(inp, axis=3)
    res_g = np.copy(res_r)
    res_b = np.copy(res_r)

    for c in range(inp.shape[-1]):
        res_r[res_r == c] =  list_of_colors[c,0]
        res_g[res_g == c] =  list_of_colors[c,1]
        res_b[res_b == c] =  list_of_colors[c,2]

    return np.concatenate([res_r[...,None],res_g[...,None],res_b[...,None]], axis=3)

if not sp_y_is_rgb:
    images_y_raw_rgb = multichannel_to_rgb(y_result)
'''

###ACTIVE CELL###
###Save data to file
Image_generator.save_data(x_result, y_result, save_dir = "../../Data/scrapid/training_data/", name="scrapid_11c_256x256_1638_noraw_1")



import cv2 as cv

from setup_machine import *

###Set enviroment. 
#On remote server script is in subfolder ./projectname/
#On my local machine script is in subfolder ./projects/projectname/, so use this to to fix the path

SETUP_PROJECT_NAME = "scrapid"
SETUP_Y_IS_RGB = False #Y is in RGB format or one channel
SETUP_NUM_OF_CLASSES = 11 #Must be fixed
SETUP_MODEL_INPUT_SHAPE = (256,256,3) #Model

#List of codes for RGB y, do not change the order!
SETUP_LIST_OF_CLASS_COLORS = [
    0, #Black
    8388608, #'HBI2': [128, 0, 0], #Dark red
    32768, #'LeichterStahlaltschrottE1': [0, 128, 0], #Green
    8421376, #'LeichterStahlneuschrottE8': [128, 128, 0], Dirty yellow
    128, #'PaketneuschrottE6': [0, 0, 128], Blue
    8388736, #'SchwererStahlaltschrottE3': [128, 0, 128], #Violet
    32896, #'SchwererStahlneuschrottE2': [0, 128, 128], #Ocean wave
    8421504, #'SpanschrottE5': [128, 128, 128], #Gray
    4194304, #'Stahlbaer': [64, 0, 0], #Very dark red
    12582912, #'unlegiert3110': [192, 0, 0], Red
    4227072, #'Verteilerbaer': [64, 128, 0], Green-yellow
    #12615680 #'unknown': [192, 128, 0] #Orange
]

#List for decoding y data. In case of one channel is [0-background,1,2....SETUP_NUM_OF_CLASSES-1]
SETUP_LIST_OF_CLASS_INDEXES = SETUP_LIST_OF_CLASS_COLORS if SETUP_Y_IS_RGB else [i for i in range(SETUP_NUM_OF_CLASSES)]

#Select opencv read mode for input images.
SETUP_CV_READ_MODE = cv.IMREAD_COLOR if SETUP_Y_IS_RGB else cv.IMREAD_UNCHANGED

#Setup dirs
SETUP_BASE_DIR = "../../data/%s/"%SETUP_PROJECT_NAME if SETUP_WORK_LOCAL else "./data/"
SETUP_PRETRAINED_MODELS_DIR = "../../data/_pretrained_models/" if SETUP_WORK_LOCAL else "./data/_pretrained_models/"


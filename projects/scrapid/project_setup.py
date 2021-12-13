import cv2 as cv

###Set enviroment. 
#On remote server script is in subfolder ./projectname/
#On my local machine script is in subfolder ./projects/projectname/, so use this to to fix the path

SETUP_WORK_LOCAL = True
SETUP_PROJECT_NAME = "scrapid"
SETUP_Y_IS_RGB = False #Y is in RGB format or one channel
SETUP_NUM_OF_CLASSES = 11 #Must be fixed
SETUP_MODEL_INPUT_SHAPE = (256,256,3) #Model


#List of codes for RGB y, do not change the order!
SETUP_LIST_OF_CLASS_COLORS = [
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

#List for decoding y data. In case of one channel is [0-background,1,2....SETUP_NUM_OF_CLASSES-1]
SETUP_LIST_OF_CLASS_INDEXES = SETUP_LIST_OF_CLASS_COLORS if SETUP_Y_IS_RGB else [i for i in range(SETUP_NUM_OF_CLASSES)]

#Select opencv read mode for input images.
SETUP_CV_READ_MODE = cv.IMREAD_COLOR if SETUP_Y_IS_RGB else cv.IMREAD_UNCHANGED

#Setup dirs
SETUP_BASE_DIR = "../../data/%s/"%SETUP_PROJECT_NAME if SETUP_WORK_LOCAL else "./data/"
SETUP_PRETRAINED_MODELS_DIR = "../../data/_pretrained_models/" if SETUP_WORK_LOCAL else "./data/_pretrained_models/"


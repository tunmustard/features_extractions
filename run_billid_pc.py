import cv2 as cv
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.distribute import parameter_server_strategy
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

import importlib
import copy

import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors

###Import self-written library
import featexlib
from featexlib import Debug as dbug
from featexlib import *


import threading
try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident
   
tf.compat.v1.enable_eager_execution()
       

###Loading models
def unet(pretrained_weights = None,input_size = (256,256,3)):
    VGG16_weight = "./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG16 = tf.keras.applications.VGG16(include_top=False, weights=VGG16_weight, input_shape=input_size)
    last_layer = VGG16.output
    
    set_trainable = False
    for layer in VGG16.layers:
        if layer.name in ['block1_conv1']:
            set_trainable = True
        if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
            layer.trainable = False
            
    model_ = tf.keras.layers.Conv2DTranspose(256,(3,3),strides=(2, 2), padding='same')(last_layer)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)    
    
    concat_1 = tf.keras.layers.concatenate([model_,VGG16.get_layer("block5_conv3").output])
    
    model_ = tf.keras.layers.Conv2D(512,(3,3),strides=(1, 1), padding='same')(concat_1)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model_ = tf.keras.layers.Conv2DTranspose(512,(3,3),strides=(2, 2),padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_) 
    
    concat_2 = tf.keras.layers.concatenate([model_,VGG16.get_layer("block4_conv3").output])
    
    model_ = tf.keras.layers.Conv2D(512,(3,3),strides=(1, 1), padding='same')(concat_2)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model_ = tf.keras.layers.Conv2DTranspose(512,(3,3),strides=(2, 2), padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_) 
    
    concat_3 = tf.keras.layers.concatenate([model_,VGG16.get_layer("block3_conv3").output])
    
    model_ = tf.keras.layers.Conv2D(256,(3,3),strides=(1, 1), padding='same')(concat_3)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model_ = tf.keras.layers.Conv2DTranspose(256,(3,3),strides=(2, 2),padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_) 
    
    concat_4 = tf.keras.layers.concatenate([model_,VGG16.get_layer("block2_conv2").output])
    
    model_ = tf.keras.layers.Conv2D(128,(3,3),strides=(1, 1), padding='same')(concat_4)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model_ = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(2, 2),padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_) 
    
    concat_5 = tf.keras.layers.concatenate([model_,VGG16.get_layer("block1_conv2").output])
    
    model_ = tf.keras.layers.Conv2D(64,(3,3),strides=(1, 1), padding='same')(concat_5)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    '''model_ = tf.keras.layers.Conv2D(32,(3,3),strides=(1, 1), padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)'''
    
    model_ = tf.keras.layers.Conv2D(1,(3,3),strides=(1, 1),padding='same')(model_)
    model_ = tf.keras.layers.LeakyReLU(0.1)(model_)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model_ = tf.keras.Model(VGG16.input,model_)
    
    model_.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
        model_.load_weights(pretrained_weights)
    
    return model_

model = unet(pretrained_weights ='./models/billid_256x256_unet_vgg16_v_01')
       
       
       



###Create process class implementation      
class Production_unit(Production):
    def __init__(self):
        
        ###Initialize process layers
        self.layer_inp = Production.Layer_input(out_channel = 1)
        self.layer_1_cut = Production.Layer_cut(t=0,l=80,h=480,w=480, out_channel = 2)
        #self.layer_2_gray = Production.Layer_to_gray(invert=True, inp_channel = 6, out_channel = 2)
        #self.layer_3_rgb = Production.Layer_to_rgb(mode="b")
        self.layer_3_resize = Production.Layer_resize(w=256,h=256, inp_channel = 2, out_channel = 2)
        #self.layer_4_normalize = Production.Layer_normalize(mean_shift=0, std_offset_pos=2, std_offset_neg=2, inp_channel = 2, out_channel = 2)
        #self.layer_5_east_model = Production.Layer_bounding_box_east(pad_h=5, pad_w=5, color_scale = -1, score_threshold = 0.5, nms_treshold = 0.4, inp_channel = 2, out_channel = 3)
        self.layer_5_scaler = Production.Layer_scaler(file="./models/scalers/billid_256x256_unet_scaler_v_test_1.pkl", inp_channel = 2, out_channel = 3, astype='float32')
        self.layer_6_model = Production.Layer_model(model=model, input_shape = (-1,256, 256, 3), output_shape = (256,256,1), inp_channel = 3, out_channel = 7)
        #self.layer_7_tresh = Production.Layer_treshold(ll=0.5, lh=1.0, inp_channel = 7, out_channel = 3)
        #self.layer_8_bbox = Production.Layer_bounding_box(pad_h=2, pad_w=2, w_roi_ct=128, w_roi_cl=128, w_k1 = 1, w_k2 = 1, w_sq = 2688, inp_channel = 3, out_channel = 3)
        #self.layer_9_cut_resize = Production.Layer_cut_bb_resize_rotate(w=84, h=32, inp_channel_1 = 2, inp_channel_2 = 3, out_channel = 4)
        #self.layer_9_1_gray = Production.Layer_to_gray(invert=True, inp_channel = 4, out_channel = 4)
        #self.layer_10_scaler = Production.Layer_scaler(file="../../models/scalers/dig4_256f_e5_cl2000_acc100_on_1000.pkl", inp_channel = 4, out_channel = 5, astype='float32')
        #self.layer_11_model = Production.Layer_model_feature(model=model_feat, input_shape = (-1,32,84,1), output_shape = (256), inp_channel = 5, out_channel = 5)
        #self.layer_show_bb = Production.Layer_show_bb(inp_channel_1 = 1, inp_channel_2 = 3, out_channel = 1)
        #self.layer_show_id = Production.Layer_add_id(id_module = Feature_detection((256,),
        #                                                 center_norm_lim=5, 
        #                                                 center_dist_norm_lim=5, 
        #                                                 likeness_lim = 3
        #                                                ), inp_channel_1 = 1, inp_channel_2 = 5, out_channel = 1)
        
        ###Initialise pipeline
        self.pipeline = Production.Pipeline_model_feat(
            proc_layers = [
                self.layer_inp,
                self.layer_1_cut,
                #self.layer_2_gray,
                self.layer_3_resize,
                #self.layer_4_normalize,
                #self.layer_5_east_model,
                self.layer_5_scaler,
                self.layer_6_model,
                #self.layer_7_tresh,
                #self.layer_8_bbox,
                #self.layer_9_cut_resize,
                #self.layer_9_1_gray,
                #self.layer_10_scaler,
                #self.layer_11_model,
                #self.layer_show_bb,
                #self.layer_show_id
            ],
            inp_channel = 1,
            out_channels = (1,2,3,7)
        )
        
        super().__init__(self.pipeline)      

production = Production_unit()
       
def gstreamer_pipeline(
    capture_width=408,
    capture_height=308,
    display_width=408,
    display_height=308,
    framerate=15,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv.VideoCapture(0)
if cap.isOpened():
    window_handle = cv.namedWindow("CSI Camera", cv.WINDOW_AUTOSIZE)
    # Window
    while cv.getWindowProperty("CSI Camera", 0) >= 0:
        ret_val, img = cap.read()
        
        img = img[...,[2,1,0]]
        
        result = production(img)[3].data
        
        #img_mask =  np.concatenate([np.zeros((256,256,1)),np.zeros((256,256,1)),result], axis=2)
        img_mask =  np.concatenate([result,result,result], axis=2)
        #img = production(img)[1].data
        
        #cv.imshow("CSI Camera", img+
        cv.imshow("CSI Camera", img_mask)
        # This also acts as
        keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()    
    cv.destroyAllWindows()
else:
    print("Unable to open camera")



#img = (np.random.rand(308,408,3)*255).astype("uint8")
#img = production(img)[0].data
#print("done", img.shape)

# Window
#window_handle = cv.namedWindow("CSI Camera", cv.WINDOW_AUTOSIZE)
#while cv.getWindowProperty("CSI Camera", 0) >= 0:
#    cv.imshow("CSI Camera", img)
    # This also acts as
#    keyCode = cv.waitKey(30) & 0xFF
    # Stop the program on the ESC key
#    if keyCode == 27:
#        break
#cv.destroyAllWindows()


            

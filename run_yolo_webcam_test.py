import cv2 as cv
import torch

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


# Model
class Model_decorator(object):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
    def __call__(self, test_image):
        width = np.max(test_image.shape)
        return self.model(test_image, size=width).render()[0]

model = Model_decorator()
###Create process class implementation      
class Production_unit(Production):
    def __init__(self):
        
        ###Initialize process layers
        self.layer_inp = Production.Layer_input(out_channel = 1)
        self.layer_1_cut = Production.Layer_cut(t=0,l=260,h=280,w=280, out_channel = 2)
        #self.layer_2_gray = Production.Layer_to_gray(invert=True, inp_channel = 6, out_channel = 2)
        #self.layer_3_rgb = Production.Layer_to_rgb(mode="b")
        self.layer_3_resize = Production.Layer_resize(w=512,h=512, inp_channel = 1, out_channel = 2)
        #self.layer_4_normalize = Production.Layer_normalize(mean_shift=0, std_offset_pos=2, std_offset_neg=2, inp_channel = 2, out_channel = 2)
        #self.layer_5_east_model = Production.Layer_bounding_box_east(pad_h=5, pad_w=5, color_scale = -1, score_threshold = 0.5, nms_treshold = 0.4, inp_channel = 2, out_channel = 3)
        #self.layer_5_scaler = Production.Layer_scaler(file="../../models/scalers/billid_256x256_unet_scaler_v_test_1.pkl", inp_channel = 2, out_channel = 3, astype='float32')
        self.layer_6_model = Production.Layer_model(model=model, input_shape = (512, 512, 3), output_shape = (512,512,3), inp_channel = 2, out_channel = 7)
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
                #self.layer_1_cut,
                #self.layer_2_gray,
                self.layer_3_resize,
                #self.layer_4_normalize,
                #self.layer_5_east_model,
                #self.layer_5_scaler,
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
            out_channels = (1,2,7)
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

        result = production(img)[2].data
        
        cv.imshow("CSI Camera", result)

        # This also acts as
        keyCode = cv.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()    
    cv.destroyAllWindows()
else:
    print("Unable to open camera")



            

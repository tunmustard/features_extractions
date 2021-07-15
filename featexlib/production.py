###Production class 
###CLASS ORIGIN IS IN NOTEBOOK, IN PROJECT BILLID, DO NOT EDIT HERE

import pickle
import numpy as np
import copy
import cv2 as cv
from .feature_detection import *


class Production(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def __call__(self, inp_data):
        return self.pipeline(inp_data)
    
    class Pipeline_base(object):
        def __init__(self, proc_layers, inp_channel = 1, out_channels = (1)):
            self.proc_layers = proc_layers
            self.inp_channel = inp_channel
            self.out_channels = out_channels
        
    class Pipeline_model_feat(Pipeline_base):
        def __init__(self, proc_layers, inp_channel = 1, out_channels = (1)):
            super().__init__(proc_layers, inp_channel = inp_channel, out_channels = out_channels)
        
        def __call__(self, inp_data):
            #Initialize inter layer object
            out_channels_process_data = {self.inp_channel:Production.Process_data(inp_data, info={})}
            
            for proc_layer in self.proc_layers:
                out_channels_process_data = proc_layer(out_channels_process_data)
            
            #return output channel
            return  [out_channels_process_data[i] for i in self.out_channels]
    
    class Layer(object):
        def __init__(self, inp_channel = 1, out_channel = 1):
            self.proc_data = {}
            self.ic = inp_channel
            self.oc = out_channel
            
        def __call__(self, channels_process_data):
            if self.oc == self.ic:
                channels_process_data[self.oc] = self.calc(channels_process_data[self.ic])
            else:
                channels_process_data[self.oc] =  self.calc(copy.deepcopy(channels_process_data[self.ic]))
            return channels_process_data
        
        ###Calc should be overwritten in subclass
        def calc(self, process_data):
            return process_data   

    class Layer_2_inputs(Layer):
        def __init__(self, inp_channel_1 = 1, inp_channel_2 = 2, out_channel = 1):
            self.ic = (inp_channel_1, inp_channel_2)
            self.oc = out_channel

        def __call__(self, channels_process_data):
            if any(list(map(lambda x: x==self.oc,self.ic))):
                channels_process_data[self.oc] = self.calc(channels_process_data[self.ic[0]],channels_process_data[self.ic[1]])
            else:
                channels_process_data[self.oc] =  self.calc(copy.deepcopy(channels_process_data[self.ic[0]]),copy.deepcopy(channels_process_data[self.ic[1]]))
            return channels_process_data
        
        ###Calc should be overwritten in subclass
        def calc(self, process_data):
            return process_data   
        
    ###Process layer: Cut part of image
    class Layer_cut(Layer):
        def __init__(self, t,l,h,w, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.t, self.l, self.h, self.w = t,l,h,w
            
        def calc(self, process_data):
            inp_img = process_data.data
            
            #Check limits
            t = self.t if self.t<=inp_img.shape[0] else inp_img.shape[0]
            l = self.l if self.l<=inp_img.shape[1] else inp_img.shape[1]
            b = self.t+self.h if (self.t+self.h)<=inp_img.shape[0] else inp_img.shape[0]
            r = self.l+self.w if (self.l+self.w)<=inp_img.shape[1] else inp_img.shape[1]
            
            #Cut image
            out_img = inp_img[t:b,l:r,...]
            
            #Store data
            process_data.data = out_img
            
            process_data.info["transform_chain"].append(("cut",(t,l,b,r)))
            
            return process_data

    ###Process layer: Input image
    class Layer_input(Layer):
        def __init__(self, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            
        def calc(self, process_data):
            inp_img = np.array(process_data.data)
            
            #Check inp shape (AxB) or (AxBxC)
            if((inp_img.ndim > 3) or (inp_img.ndim < 2)):
                raise ValueError("Wrong image dimension")
            
            #Reshape (AxB)->(AxBx1)
            if (inp_img.ndim==2):
                inp_img = inp_img[...,None]
            
            #Add info and data
            process_data.data = inp_img
            process_data.info["transform_chain"] = []
            process_data.info["inp_shape"] = inp_img.shape
            return process_data
        
    ###Process layer: Convert to gray
    class Layer_to_gray(Layer):
        def __init__(self, rgb_weights = [0.2989, 0.5870, 0.1140], invert = False, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.rgb_weights = rgb_weights
            self.invert = invert
            
        def calc(self, process_data):
            if 'fail' in process_data.info:
                return process_data
            process_data.data = (255-np.dot(process_data.data[...,:3], self.rgb_weights) if self.invert else np.dot(process_data.data[...,:3], self.rgb_weights))[...,None].astype(np.uint8)
            return process_data  

    ###Process layer: Convert gray to R or G or B
    class Layer_to_rgb(Layer):
        def __init__(self, mode = "r", inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.mode = mode
            
        def calc(self, process_data):
            out_shape = list(process_data.data.shape)
            out_shape[-1] = 2
            
            out = np.concatenate((process_data.data, np.zeros(out_shape,dtype=np.uint8)), axis=2)
            if self.mode == "b":
                process_data.data = out
            if self.mode == "g":
                process_data.data = out[...,[1,0,2]]
            if self.mode == "r":
                process_data.data = out[...,[1,2,0]]
            
            process_data.info["color"] = self.mode
            
            return process_data  

    ###Process layer: resize
    class Layer_resize(Layer):
        def __init__(self, w, h, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.w, self.h = w, h
            
        def calc(self, process_data):
            process_data.info["transform_chain"].append(("resize",(process_data.data.shape[1]/self.w,process_data.data.shape[0]/self.h)))
            process_data.data = cv.resize(process_data.data, (self.w,self.h), interpolation = cv.INTER_AREA) 
            return process_data 
        
    ###Process layer: scaler
    class Layer_scaler(Layer):
        def __init__(self, file="", inp_channel = 1, out_channel = 1 , astype=None):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.scaler = pickle.load(open(file,'rb'))
            self.astype = astype
            
        def calc(self, process_data):
            
            if 'fail' in process_data.info:
                return process_data
            
            inp_shape = process_data.data.shape
            process_data.data = (self.scaler.transform(process_data.data.reshape(-1,np.prod(inp_shape)))).reshape(inp_shape)

            if self.astype is not None:
                process_data.data = process_data.data.astype(self.astype)
            return process_data 
        
    ###Process layer: model
    class Layer_model(Layer):
        def __init__(self, model, input_shape, output_shape=None, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.model = model
            self.input_shape = input_shape
            self.output_shape = output_shape
            
        def calc(self, process_data):
            
            if 'fail' in process_data.info:
                return process_data
            
            result = self.model(process_data.data.reshape(self.input_shape)).numpy()
            
            if self.output_shape is not None:
                result = result.reshape(self.output_shape)
            else:
                result = result.reshape(self.input_shape[1:])
                
            process_data.data = result
            return process_data
        
    ###Process layer: model for feature extraction
    class Layer_model_feature(Layer):
        def __init__(self, model, input_shape, output_shape=None, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.model = model
            self.input_shape = input_shape
            self.output_shape = output_shape
            
        def calc(self, process_data):
            if 'fail' in process_data.info:
                return process_data
            
            _, features = self.model(process_data.data.reshape(self.input_shape))
            
            if self.output_shape is not None:
                result = features.numpy().reshape(self.output_shape)
            else:
                result = features.numpy().reshape(self.input_shape[1:])
                
            process_data.data = result
            return process_data
        
    ###Process layer: normalize (-STD*k|<--mean(+/-)-->|STD*k)->Normalize
    class Layer_normalize(Layer):
        def __init__(self, mean_shift=0, std_offset_pos=1, std_offset_neg=1, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            
            self.mean_shift, self.std_offset_pos, self.std_offset_neg = mean_shift, std_offset_pos, std_offset_neg
            
        def calc(self, process_data):
            mean, STD  = cv.meanStdDev(process_data.data)
            process_data.data = np.clip(process_data.data + (mean*self.mean_shift), mean*(1+self.mean_shift) - self.std_offset_neg*STD, mean*(1+self.mean_shift) + self.std_offset_pos*STD )# 
            process_data.data = np.clip(process_data.data, 0, 255)
            process_data.data = cv.normalize(process_data.data, None, 0, 255, norm_type=cv.NORM_MINMAX)
            return process_data
    
    ###Process layer: treshold
    class Layer_treshold(Layer):
        def __init__(self, ll=0.5, lh=1.0, info_id = 1, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.ll = ll
            self.lh = lh
            self.info_id = info_id
            
        def calc(self, process_data):
            process_data.data = cv.threshold(process_data.data, self.ll, self.lh, cv.THRESH_BINARY)[1].astype('uint8')
            return process_data 
        
    ###Process layer: get bounding box
    class Layer_bounding_box(Layer):
        def __init__(self, pad_h=2, pad_w=2, w_roi_ct=128, w_roi_cl=128, w_k1 = 1, w_k2 = 1, w_sq = 2688, inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.pad_h = pad_h
            self.pad_w = pad_w
            self.w_roi_ct = w_roi_ct
            self.w_roi_cl = w_roi_cl
            self.w_k1 = w_k1
            self.w_k2 = w_k2
            self.w_sq = w_sq
            
        def calc(self, process_data):
            contours, hierarchy = cv.findContours(process_data.data.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            bb_list = [cv.boundingRect(contour) for contour in contours]
            
            pic_h = process_data.data.shape[0]
            pic_w = process_data.data.shape[1]
            pic_diag = np.sqrt(pic_h**2+pic_w**2)
            
            #Find best bounding box
            if(bb_list):
                mp_find = []
                for bb in bb_list:
                    cen_l = (bb[0]+bb[2]/2)
                    cen_t = (bb[1]+bb[3]/2)
                    #Weight function to find best contour
                    mp_find.append((np.sqrt((self.w_roi_ct-cen_t)**2+(self.w_roi_cl-cen_l)**2)/pic_diag)*self.w_k1 + self.w_k2*(self.w_sq/(bb[2]*bb[3])))
                #result = cv.drawContours(result, contours[0], -1, 255, 3)
                bb = bb_list[np.argmin(mp_find)]
                cw = int((bb[0]+bb[2]/2))
                ch = int((bb[1]+bb[3]/2))
                half_size = int(np.max([bb[2],bb[3]])/2)
                boards_lr = np.clip([cw-half_size-self.pad_w, cw+half_size+self.pad_w], 0, pic_w-1)
                boards_tb = np.clip([ch-half_size-self.pad_h, ch+half_size+self.pad_h], 0, pic_h-1)
                t,l,b,r = boards_tb[0],boards_lr[0],boards_tb[1],boards_lr[1]
                
                process_data.info["bb"] = (t,l,b,r)
                process_data.data = cv.rectangle(process_data.data*255, (l, t), (r, b), 255, 4)
            return process_data 

    ###Process layer: get bounding from EAST model opencv
    class Layer_bounding_box_east(Layer):
        def __init__(self, file="models/frozen_east_text_detection.pb", model_width=160, model_height=160, pad_h=2, pad_w=2, color_scale = 1, score_threshold = 0.5, nms_treshold = 0.4, out_layers=["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"], inp_channel = 1, out_channel = 1):
            super().__init__(inp_channel = inp_channel, out_channel = out_channel)  
            self.pad_h = pad_h
            self.pad_w = pad_w
            self.score_threshold = score_threshold
            self.nms_treshold = nms_treshold
            self.outputLayers = out_layers
            self.model = cv.dnn.readNet(file)
            self.color_scale = color_scale
            self.model_width = model_width
            self.model_height = model_height
            
        def decodeBoundingBoxes(self, scores, geometry, scoreThresh):
            detections = []
            confidences = []

            ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
            assert len(scores.shape) == 4, "Incorrect dimensions of scores"
            assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
            assert scores.shape[0] == 1, "Invalid dimensions of scores"
            assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
            assert scores.shape[1] == 1, "Invalid dimensions of scores"
            assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
            assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
            assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
            height = scores.shape[2]
            width = scores.shape[3]
            for y in range(0, height):

                # Extract data from scores
                scoresData = scores[0][0][y]
                x0_data = geometry[0][0][y]
                x1_data = geometry[0][1][y]
                x2_data = geometry[0][2][y]
                x3_data = geometry[0][3][y]
                anglesData = geometry[0][4][y]
                for x in range(0, width):
                    score = scoresData[x]

                    # If score is lower than threshold score, move to next x
                    if (score < scoreThresh):
                        continue

                    # Calculate offset
                    offsetX = x * 4.0
                    offsetY = y * 4.0
                    angle = anglesData[x]

                    # Calculate cos and sin of angle
                    cosA = np.cos(angle)
                    sinA = np.sin(angle)
                    h = x0_data[x] + x2_data[x]
                    w = x1_data[x] + x3_data[x]

                    # Calculate offset
                    offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                    # Find points for rectangle
                    p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                    p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                    center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                    detections.append((center, (w, h), -1 * angle * 180.0 / np.pi))
                    confidences.append(float(score))

            # Return detections and confidences
            return [detections, confidences]
        
        def calc(self, process_data):
            
            blob = cv.dnn.blobFromImage(process_data.data, self.color_scale, (self.model_width, self.model_height), (123.68, 116.78, 103.94), True, True)

            self.model.setInput(blob)
            output = self.model.forward(self.outputLayers)
            scores = output[0]
            geometry = output[1]
            
            h = process_data.data.shape[0]
            w = process_data.data.shape[1]

            process_data.data = cv.resize((scores.reshape((scores.shape[2],scores.shape[3],1))*255).astype(np.uint8), (w,h), interpolation = cv.INTER_AREA) 
            
            rW = w / float(self.model_width)
            rH = h / float(self.model_height)

            [rects, confidences] = self.decodeBoundingBoxes(scores, geometry, self.score_threshold)

            indices = cv.dnn.NMSBoxesRotated(rects, confidences, self.score_threshold, self.nms_treshold)
            
            if indices:
                #Take first indice
                i = indices[0]

                coords = rects[i[0]]

                # get 4 corners of the rotated rect
                vertices = cv.boxPoints(coords)

                # scale the bounding box coordinates based on the respective ratios
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                    cv.circle(process_data.data, (vertices[j][0], vertices[j][1]), 4, (0,0,255), 1)



                cl,ct = int(coords[0][0]*rW),int(coords[0][1]*rH)
 
                cv.circle(process_data.data, (cl,ct), 1, (0,0,255), 1)
                cw,ch = int(coords[1][0]*rW),int(coords[1][1]*rH)
                
                size = np.max([cw,ch])
                
                t,l = int(ct-size/2)-self.pad_h, int(cl-size/2)-self.pad_w
                b,r = t+size+self.pad_h, l+size+self.pad_w

                process_data.info["bb"] = (t,l,b,r)
                process_data.info["bb_center"] = (cl,ct)
                process_data.info["bb_angle"] = coords[2]
                
                process_data.data = cv.rectangle(process_data.data, (l, t), (r, b), 255, 4)

            return process_data 
        
    ###Process layer: cut image by bounding box
    class Layer_cut_bb_resize_rotate(Layer_2_inputs):
        def __init__(self, w=84, h=32, inp_channel_1 = 1, inp_channel_2 = 2, out_channel = 1):
            super().__init__(inp_channel_1 = inp_channel_1, inp_channel_2 = inp_channel_2, out_channel=out_channel)
            self.w = w
            self.h = h
     
        def calc(self, process_data_1, process_data_2):

            if "bb" in process_data_2.info:
                t,l,b,r = process_data_2.info["bb"][0],process_data_2.info["bb"][1],process_data_2.info["bb"][2],process_data_2.info["bb"][3]

                pic_h = process_data_1.data.shape[0]
                pic_w = process_data_1.data.shape[1]
                if(process_data_1.data.ndim>2):
                    channels = process_data_1.data.shape[2]
                else:
                    channels = 1

                if "bb_angle" in process_data_2.info:
                    angle = process_data_2.info["bb_angle"]
                    (cl,ct) = process_data_2.info["bb_center"] 
                    w,h = r-l, b-t 
                    
                    rad = int(np.sqrt(((h)**2) + ((w)**2))/2)

                    rot_lr = np.clip([cl-rad, cl+rad], 0, pic_w-1)
                    rot_tb = np.clip([ct-rad, ct+rad], 0, pic_h-1)
                    rot_t,rot_l,rot_b,rot_r = rot_tb[0],rot_lr[0],rot_tb[1],rot_lr[1]

                    img_cut = process_data_1.data[rot_t:rot_b,rot_l:rot_r]
                    

                    img_bg = np.zeros((rad*2,rad*2,channels),dtype=np.uint8)

                    offsets = np.clip([rad-ct,rad-cl], 0, rad*2)
                    offset_h,offset_w = offsets[0],offsets[1]

                    img_bg[offset_h:offset_h+img_cut.shape[0],offset_w:offset_w+img_cut.shape[1]] = img_cut 

                    M = cv.getRotationMatrix2D((rad,rad), angle, 1.0)
                    img_rotated = cv.warpAffine(img_bg, M, (img_bg.shape[1],img_bg.shape[0]), flags=cv.INTER_LINEAR)

                    cut_offset_t = int((2*rad-h)/2) 
                    cut_offset_l = int((2*rad-w)/2) 

                    #print(cut_offset_t,cut_offset_t+h,cut_offset_l,cut_offset_l+w)
                    process_data_1.data = img_rotated[cut_offset_t:cut_offset_t+h,cut_offset_l:cut_offset_l+w]
                else:    
                    #Cut image
                    process_data_1.data = process_data_1.data[t:b,l:r]
                    
                process_data_1.info["transform_chain"].append(("cut",(t,l,b,r)))

                size = np.max([self.h,self.w])
                diff_h = int((size - self.h)/2)
                diff_w = int((size - self.w)/2)
                
                process_data_1.data = cv.resize(process_data_1.data, (size,size), interpolation = cv.INTER_AREA)[diff_h:diff_h+self.h,diff_w:diff_w+self.w] 
                process_data_1.info["bb_cut"] = (self.h,self.w)
                process_data_1.info.update(process_data_2.info)
            else:
                process_data_1.data = None
                process_data_1.info['fail'] = True
            
            return process_data_1 

    ###Process layer: show bb on input image
    class Layer_show_bb(Layer_2_inputs):
        def __init__(self, inp_channel_1 = 1, inp_channel_2 = 2, out_channel = 1, id_module = Feature_detection((256,))):
            super().__init__(inp_channel_1 = inp_channel_1, inp_channel_2 = inp_channel_2, out_channel=out_channel)
            self.id_module = id_module
             
        def calc(self, process_data_1, process_data_2):
            if 'fail' in process_data_2.info or "bb" not in process_data_2.info:
                return process_data_1
            
            np_tlrb = np.array(process_data_2.info["bb"])

            for i in reversed(process_data_2.info["transform_chain"]):
                if i[0]=="resize":
                    (resize_t, resize_l) = i[1]
                    np_tlrb = np_tlrb*np.array([resize_t, resize_l, resize_t, resize_l])

                elif i[0]=="cut":
                    (cut_t, cut_l, _, _) = i[1]
                    np_tlrb = np_tlrb+np.array([cut_t,cut_l,cut_t,cut_l])
            
            np_tlrb = np_tlrb.astype(np.uint)
            t,l,b,r =  np_tlrb[0],np_tlrb[1],np_tlrb[2],np_tlrb[3]     
            process_data_1.data = cv.rectangle(process_data_1.data, (l, t), (r, b), 255, 4)
            process_data_1.info["bb_abs"] = (t,l,b,r)
            process_data_1.info.update(process_data_2.info)
            return process_data_1
        
    class Layer_add_id(Layer_2_inputs):
        def __init__(self, id_module = Feature_detection((256,),
                                                         center_norm_lim=5, 
                                                         center_dist_norm_lim=5, 
                                                         likeness_lim = 3
                                                        ), inp_channel_1 = 1, inp_channel_2 = 2, out_channel = 1):
            super().__init__(inp_channel_1 = inp_channel_1, inp_channel_2 = inp_channel_2, out_channel=out_channel)
            self.id_module = id_module
             
        def calc(self, process_data_1, process_data_2):
            if self.id_module is not None:
                num_id = self.id_module(process_data_2.data)
                print("num_id = ",num_id)
                if "fail" not in process_data_2.info and num_id is not None:
                    (t,l,b,r) = process_data_1.info["bb_abs"]
                    t = int(t+50)
                    l = int(l+10)
                    font = cv.FONT_HERSHEY_DUPLEX
                    cv.putText(process_data_1.data, "%s"%num_id, (l, t), font, 2.0, (0, 255, 0), 3)

            return process_data_1
        
    #Base class for transfer data between layers
    class Process_data(object):
        def __init__(self, data, info={}):
            self.data = data
            self.info = info
###Specian class for image transformation and augmentation
###CLASS ORIGIN IS HERE
#COLORS ORDER IS 'Blue-Green-Red, as in openCV


import os
import numpy as np
import cv2 as cv
import re
import glob
import pickle
from math import pi

class Image_generator(object):
    def __init__(self, pipeline):
        self.num_classes = None
        self.num_class_samples = None
        self.pipeline = pipeline
        
    def __call__(self, *args, shuffle = False):
        #Execute pipeline
        pl1_x, pl1_y = self.pipeline(*args)
        
        #Set some data
        self.num_classes = self.pipeline.num_classes
        self.num_class_samples = self.pipeline.num_class_samples
        
        #Output x,y arrays as is or reshuffled
        if shuffle:
            pl1_x, pl1_y = self.shuffle_x_y(pl1_x, pl1_y)
        
        return pl1_x, pl1_y 
    
    ###Pipeline base classes
    class Pipeline(object):
        def __init__(self, common_layers, special_layers):
            self.common_layers = common_layers
            self.special_layers = special_layers
            self.num_classes = None
            self.num_class_samples = None
        
    ###Pipeline with labeles outputs
    class Pipeline_labels(Pipeline):
        def __init__(self, common_layers, special_layers):
            super().__init__(common_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict, target="all")
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                out_trans_dict = inp_trans_dict.copy()
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict, target="x")
                    
                result_x.append(out) 
                result_y.append(np.array([i for i in range(inp_common.shape[0])]))
                del out
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = result_y.max()+1
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            return result_x, result_y
    
    ###Pipeline with angles output
    class Pipeline_angles(Pipeline):
        def __init__(self, common_layers, special_layers):
            super().__init__(common_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict, target="all")
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                out_trans_dict = inp_trans_dict.copy()
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict, target="x")
                    
                result_x.append(out) 
                result_y.append(out_trans_dict["sin_cos_tan"])
                del out, out_trans_dict
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = len(np.unique(result_y))
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            return result_x, result_y  

    ###Pipeline with angles output
    class Pipeline_bb_angle(Pipeline):
        def __init__(self, common_layers, special_layers):
            super().__init__(common_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict, target="all")
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                out_trans_dict = inp_trans_dict.copy()
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict, target="x")
                    
                result_x.append(out) 
                
                y_add = np.append(out_trans_dict["bbox"], out_trans_dict["sin_cos_tan"], axis=1)
                result_y.append(y_add)
                del out, out_trans_dict
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = len(np.unique(result_y))
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            return result_x, result_y  
        
    ###Pipeline with marked rectangles as target
    class Pipeline_rectangles(Pipeline):
        def __init__(self, common_layers, special_layers):
            super().__init__(common_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict, target="x")
            
            result_x = []
            result_y = [] 
            
            #Process Y output only
            out_mask_common = inp
            for i in self.common_layers:
                out_mask_common, _ = i(out_mask_common, inp_trans_dict, target="y")
            
            #Generate X and Y images output
            for p in self.special_layers:
                out = np.copy(inp_common)
                out_mask = np.copy(out_mask_common)
                out_trans_dict = inp_trans_dict.copy()
                
                num_classes = 0
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict, target="x")
                    out_mask, _ = i(out_mask, out_trans_dict, target="y", use_saved = True)
                
                result_x.append(out) 
                result_y.append(out_mask)
                del out, out_trans_dict, out_mask 
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = len(inp_common)
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            del inp_common, out_mask_common
            
            return result_x, result_y  

    ###Pipeline for source image dataset (x) and target (y) image dataset
    class Pipeline_x_y_images(Pipeline):
        def __init__(self, common_layers=[], special_layers=[[]]):
            super().__init__(common_layers, special_layers)
        def __call__(self, inp_x, inp_y):
            inp_x_common = inp_x
            inp_y_common = inp_y
            inp_trans_x_dict = {}
            inp_trans_y_dict = {}
            
            for i in self.common_layers:
                inp_x_common, inp_trans_x_dict = i(inp_x_common, inp_trans_x_dict, target="x")
                inp_y_common, inp_trans_y_dict = i(inp_y_common, inp_trans_y_dict, target="y")
            
            result_x = []
            result_y = [] 
            
            #Generate X and Y images output
            for p in self.special_layers:
                out_x = np.copy(inp_x_common)
                out_y = np.copy(inp_y_common)
                out_trans_dict = inp_trans_x_dict.copy()
                
                for i in p:
                    out_x, out_trans_dict = i(out_x, out_trans_dict, target="x")
                    out_y, _ = i(out_y, out_trans_dict, target="y", use_saved = True)
                
                result_x.append(out_x) 
                result_y.append(out_y)
                del out_x, out_y, out_trans_dict 
            
            result_x = np.concatenate(result_x, axis=0)
            result_y = np.concatenate(result_y, axis=0)
            
            self.num_classes = 1
            self.num_class_samples = int(result_x.shape[0])
            
            del inp_x_common, inp_y_common
            
            return result_x, result_y 
        
    ###Pipeline parallel calculation for merging of several pipelines into one, with different xN,yN pairs
    class Pipeline_x_y_parallel(Pipeline):
        def __init__(self, pipelines_list, common_layers = [], special_layers = []):
            super().__init__(common_layers, special_layers)
            self.pipelines_list = pipelines_list
        def __call__(self, data_list):
            #Check if pipelei is correctly configurated
            assert len(data_list) == len(self.pipelines_list), "Split size of the image is too big"

            result_x = []
            result_y = [] 

            #data_list has form [[x0,y0],[x1,y1]...,[xN,yN]]
            for i,v in enumerate(data_list):
                #x,y <=> v[0],v[1]
                if v[0] is not None and len(v[0])>0:
                    x_, y_ = self.pipelines_list[i](v[0],v[1])
                    result_x.append(x_)
                    result_y.append(y_)
            
            return np.concatenate(result_x, axis=0), np.concatenate(result_y, axis=0) 

    ###Pipeline sequential calculation for combining of several pipelines, as a chain
    class Pipeline_x_y_sequential(Pipeline):
        def __init__(self, pipelines_list, common_layers = [], special_layers = []):
            super().__init__(common_layers, special_layers)
            self.pipelines_list = pipelines_list
        def __call__(self, *arg):

            #Special trick, in case if first pipeline is parallel
            inp_x, inp_y = self.pipelines_list[0](*arg)

            for i in range(1, len(self.pipelines_list)):
                inp_x, inp_y = self.pipelines_list[i](inp_x,inp_y)
            
            return inp_x, inp_y 

    #Base class for all tranformation layers
    class Mod(object):
        ###layer can aplies to 'x' image generation or 'y' image generation or "all"
        def __init__(self, target="all"):
            self.trans_dict = {}
            self.target = target #"x", "y", "all"
        def __call__(self, inp, trans_dict, target, use_saved=False):
            if self.target==target or self.target=="all":
                result, trans_dict = self.calc(inp, trans_dict, use_saved)
            else:
                result = inp
            return result, trans_dict
        ###Calc should be overwritten in subclass
        def calc(self, inp, trans_dict, use_saved=False):
            return inp, trans_dict     

    ###Preprocess layer: scaler
    class Mod_scaler(Mod):
        def __init__(self, file="", astype=None, target="x"):
            super().__init__(target=target) 
            self.scaler = pickle.load(open(file,'rb'))
            self.astype = astype
        def calc(self, inp, trans_dict, use_saved=False):
            inp_shape = inp.shape
            inp = (self.scaler.transform(inp.reshape(inp.shape[0],-1))).reshape(inp_shape)
            self.trans_dict["scaler"] = True
            trans_dict.update(self.trans_dict)
            if self.astype is not None:
                inp = inp.astype(self.astype)

            return inp, trans_dict 

    ###Preprocess layer: astype
    class Mod_astype(Mod):
        def __init__(self, astype, target="all"):
            super().__init__(target=target) 
            self.astype = astype
        def calc(self, inp, trans_dict, use_saved=False):
            self.trans_dict["astype"] = self.astype
            return inp.astype(self.astype), trans_dict 

    #Combine input digits to one long image
    class Mod_combinator(Mod):
        def __init__(self, num_digits, delete_side_margin = 0, target="all"):
            super().__init__(target=target)
            self.delete_side_margin = delete_side_margin
            self.num_digits = num_digits
        def calc(self, inp, trans_dict, use_saved=False):
            #expect input matrix with shape (-1, num_samples, n, m)
            #where n*m - image
            shape = inp.shape[2]
            self.trans_dict["num_combined"] = 4
            trans_dict.update(self.trans_dict)
            return np.concatenate([np.delete(inp[:,i,:,:], list(((k,shape-1-k) for k in range(self.delete_side_margin))), 2) for i in range(self.num_digits)], axis=2), trans_dict
        
    #Add padding left, right, top, bottom
    class Mod_add_padding(Mod):
        def __init__(self, pl=2, pr=2, pt=2, pb=2, target="all"):
            super().__init__(target=target)
            self.padd = (pl,pr,pt,pb)
        def calc(self, inp, trans_dict, use_saved=False):
            #expect input matrix with shape (-1, a, b,...)
            self.trans_dict["extra_padding"] = self.padd
            trans_dict.update(self.trans_dict)
            return np.pad(inp,[(0,0),(self.padd[2],self.padd[3]),(self.padd[0],self.padd[1])] + [(0,0) for k in range(len(inp.shape)-3)],"constant"), trans_dict

    #Resize images
    class Mod_resize(Mod):
        def __init__(self, size, target="all", interpolation = cv.INTER_AREA, interpolation_saved = None):
            super().__init__(target=target)
            self.size = size
            self.interpolation = interpolation
            self.interpolation_saved = interpolation if interpolation_saved is None else interpolation_saved
        def calc(self, inp, trans_dict, use_saved=False):
            self.trans_dict["resize"] = self.size 
            trans_dict.update(self.trans_dict)
            
            interpolation = self.interpolation_saved if use_saved else self.interpolation

            return np.concatenate([[cv.resize(i, self.size, interpolation = interpolation)] for i in inp],axis=0), trans_dict
        

    #Reshape images
    ###e.g. shape=(-1, 1071, 3557, 1) or shape=(...,None)
    class Mod_reshape(Mod):
        def __init__(self, shape, target="all"):
            super().__init__(target=target)
            self.shape = shape
        def calc(self, inp, trans_dict, use_saved=False):
            if ... in self.shape:
                return inp[self.shape], trans_dict
            else:
                return inp.reshape(self.shape), trans_dict

    #Split images in subimage
    class Mod_split(Mod):
        def __init__(self, size, target="all"):
            super().__init__(target=target)
            self.split_size_x, self.split_size_y = size[0], size[1]

        def calc(self, inp, trans_dict, use_saved=False):
            c_x_split_num = inp.shape[1]//self.split_size_x
            c_y_split_num = inp.shape[2]//self.split_size_y

            assert c_x_split_num>0 and c_y_split_num>0, "Split size of the image is too big"
            self.trans_dict["split"] = [self.split_size_x, self.split_size_y]
            trans_dict.update(self.trans_dict)

            split_result = np.split(inp, [(i+1)*self.split_size_x for i in range(c_x_split_num)], axis=1)[:c_x_split_num]
            split_result = np.concatenate(split_result,axis=0)
            split_result = np.split(split_result, [(i+1)*self.split_size_y for i in range(c_y_split_num)], axis=2)[:c_y_split_num]
            split_result = np.concatenate(split_result,axis=0)
            return split_result, trans_dict

    #Dulicate images N times
    class Mod_duplicate(Mod):
        def __init__(self, num=2, target="all"):
            super().__init__(target=target)
            self.num = num

        def calc(self, inp, trans_dict, use_saved=False):

            self.trans_dict["duplicate"] = self.num
            trans_dict.update(self.trans_dict)
            
            duplicate_result = inp

            for i in range(self.num-1):
                duplicate_result = np.concatenate([duplicate_result,duplicate_result], axis=0)

            return duplicate_result, trans_dict

    #Crop image
    class Mod_crop(Mod):
        def __init__(self, t, l, h, w, target="all"):
            super().__init__(target=target)
            self.t, self.l, self.h, self.w = t,l,h,w
        def calc(self, inp, trans_dict, use_saved=False):
            self.trans_dict["crop"] = [self.t, self.l, self.h, self.w]
            trans_dict.update(self.trans_dict)
            
            b=self.t+self.h if self.t+self.h<=inp.shape[1] else inp.shape[1]
            r=self.l+self.w if self.l+self.w<=inp.shape[2] else inp.shape[2]
            
            return inp[:,self.t:b,self.l:r,...], trans_dict

    #Perspective transformation
    class Mod_linear_transf(Mod):
        ###l,r,t,b percentage of shrinking in corrersponding side (left, right, top, bottom)
        def __init__(self, l=0.0, r=0.0, t = 0.0, b = 0.0, rand = False, target="all"):
            super().__init__(target=target)
            self.inp_sc = (l/2,r/2,t/2,b/2)
            self.rand = rand
        def calc(self, inp, trans_dict, use_saved=False):
            if not use_saved:
                self.sc = self.inp_sc*np.random.rand(4) if self.rand else self.inp_sc
            w,h = inp.shape[2],inp.shape[1]
            tl, tr, bl, br = [0,0],[w-1,0],[0,h-1],[w-1,h-1]
            pts1 = np.float32([tl,tr,bl,br])
            pts2 = np.float32(
                np.floor([
                    [self.sc[2]*w, self.sc[0]*h],
                    [w-1-self.sc[2]*w, self.sc[1]*h],
                    [self.sc[3]*w, h-1-self.sc[0]*h],
                    [w-1-self.sc[3]*w, h-1-self.sc[1]*h]
                ])
            )
            M = cv.getPerspectiveTransform(pts1,pts2)
            self.trans_dict["perspective_transform"] = self.inp_sc*2
            trans_dict.update(self.trans_dict)
            return np.array([cv.warpPerspective(i,M,(w,h)) for i in inp]), trans_dict

    #Rotate
    class Mod_rotate(Mod):
        ###angle, random
        def __init__(self, angle=20.0, rand = False, target="all", interpolation = cv.INTER_AREA, interpolation_saved = None, borderValue=0, borderValue_saved=None):
            super().__init__(target=target)
            self.in_angle = angle
            self.rand = rand
            self.borderValue = borderValue
            self.borderValue_saved = borderValue if borderValue_saved is None else borderValue_saved
            self.interpolation = interpolation
            self.interpolation_saved = interpolation if interpolation_saved is None else interpolation_saved
        def calc(self, inp, trans_dict, use_saved=False):
            w,h = inp.shape[2],inp.shape[1]
            image_center = (w/2,h/2)
            
            interpolation = self.interpolation_saved if use_saved else self.interpolation
            borderValue = self.borderValue_saved if use_saved else self.borderValue

            if not use_saved:
                self.saved_angle = [self.in_angle if not self.rand else self.in_angle * (np.random.rand(1)*2 - 1) for i in inp]
                self.shift_matrix = [cv.getRotationMatrix2D(image_center, int(self.saved_angle[num]), 1.0) for num,i in enumerate(inp)]
            
            sin = (np.sin(2*pi*np.array(self.saved_angle)/360)+1)/2
            cos = (np.cos(2*pi*np.array(self.saved_angle)/360)+1)/2
            tan = np.tan(2*pi*np.array(self.saved_angle)/360)
            self.trans_dict["sin_cos_tan"] = np.concatenate([sin,cos,tan], axis=1)
            trans_dict.update(self.trans_dict)
            return np.array([cv.warpAffine(i, self.shift_matrix[num], (w,h), flags=interpolation, borderValue = borderValue) for num,i in enumerate(inp)]), trans_dict
        
    #Shift
    class Mod_shift(Mod):
        ###sh_h, sh_v
        def __init__(self, sh_h=20, sh_v=10, rand = False, target="all", interpolation = cv.INTER_AREA, interpolation_saved = None, borderValue=0, borderValue_saved=None):
            super().__init__(target)
            self.sh_h = sh_h
            self.sh_v = sh_v
            self.rand = rand
            self.borderValue = borderValue
            self.borderValue_saved = borderValue if borderValue_saved is None else borderValue_saved
            self.interpolation = interpolation
            self.interpolation_saved = interpolation if interpolation_saved is None else interpolation_saved
        def calc(self, inp, trans_dict, use_saved=False):
            w,h = inp.shape[2],inp.shape[1]
            image_center = (w/2,h/2)
            self.trans_dict["shift"] = (self.sh_h, self.sh_v, self.rand)
            trans_dict.update(self.trans_dict)
            
            interpolation = self.interpolation_saved if use_saved else self.interpolation
            borderValue = self.borderValue_saved if use_saved else self.borderValue

            if not use_saved:
                self.shift_matrix = np.float32([[[1,0, self.sh_h * (2*np.random.rand(1)-1) if self.rand else self.sh_h],
                                            [0,1, self.sh_v * (2*np.random.rand(1)-1) if self.rand else self.sh_v]] for i in inp])

            return np.array(
                [cv.warpAffine(i, self.shift_matrix[num] ,(w,h),flags=interpolation,  borderValue = borderValue) for num, i in enumerate(inp)]
            ), trans_dict

    #Create recangle
    class Mod_add_rectangle(Mod):
        def __init__(self, t=5.0, l=5.0, w=50, h=30, color=255, rand = False, target="y"):
            super().__init__(target=target)
            self.t = t
            self.l = l
            self.w = w
            self.h = h
            self.rand = rand 
            self.color = color
        def calc(self, inp, trans_dict, use_saved=False):    
            yy, xx = np.mgrid[:inp.shape[1], :inp.shape[2]]
            xx = np.repeat(xx[None,...], [inp.shape[0]], axis=0)
            yy = np.repeat(yy[None,...], [inp.shape[0]], axis=0)
            xx[xx<self.l]=0
            xx[xx>(self.l + self.w)]=0
            yy[yy<self.t]=0
            yy[yy>(self.t + self.h)]=0
            mask=xx*yy
            inp[mask>0] = self.color
            self.trans_dict["rectangle"] = (self.t, self.l, self.w, self.h)
            trans_dict.update(self.trans_dict)
            return inp, trans_dict 

    #Create recangle
    class Mod_add_bounding_box_info(Mod):
        #Bounding box padding t,l,b,r
        def __init__(self, t=2.0, l=2.0, b=2.0, r=2.0, target="all"):
            super().__init__(target=target)
            self.t = t
            self.l = l
            self.b = b
            self.r = r
        def calc(self, inp, trans_dict, use_saved=False):  
            h=inp.shape[1]
            w=inp.shape[2]
            add_padding = np.repeat(np.array([[-self.t,-self.l,self.b,self.r]]), [len(inp)], axis=0)
            normalize = np.array([1/h,1/w,1/h,1/w])
            bbox = (np.vstack([np.concatenate((np.argwhere(i>0).min(0),np.argwhere(i>0).max(0))) for i in inp])+add_padding)*normalize
            self.trans_dict["bbox"] = bbox
            trans_dict.update(self.trans_dict)
            return inp, trans_dict 
        
    #Cut random circle in image
    class Mod_round_cut(Mod):
        def __init__(self, r = 10.0, rand_r = False, target="x"):
            super().__init__(target=target)
            self.rand_r = rand_r
            self.r = r
            
        def calc(self, inp, trans_dict, use_saved=False):     
            yy, xx = np.mgrid[:inp.shape[1], :inp.shape[2]]
            xx = np.repeat(xx[None,...], [inp.shape[0]], axis=0)
            yy = np.repeat(yy[None,...], [inp.shape[0]], axis=0)

            c_x = inp.shape[2] * np.random.rand(inp.shape[0])
            c_x = np.repeat(c_x[...,None,None], inp.shape[1], axis=1)
            c_x = np.repeat(c_x, inp.shape[2], axis=2)
            c_y = inp.shape[1] * np.random.rand(inp.shape[0])
            c_y = np.repeat(c_y[...,None,None], inp.shape[1], axis=1)
            c_y = np.repeat(c_y, inp.shape[2], axis=2)

            circle = (xx - c_x) ** 2 + (yy - c_y) ** 2
            radius = self.r * np.random.rand(1) if self.rand_r else self.r
            mask = circle > (radius**2)
            self.trans_dict["round_cut"] = (radius)
            trans_dict.update(self.trans_dict)
            return inp*mask, trans_dict
        
    #Add noise and bg
    class Mod_add_noise(Mod):
        #limiting "max" or "norm" merging "mask" or "full" or "max"
        def __init__(self, target="x", level=255, std=0.15, mean=0.0, ll=0.0, hl=0.3, inp_scale=None, k_ref=None, lim="max", merge="full", bg = None):  
            super().__init__(target=target)
            self.level = level
            self.std = std
            self.mean = mean
            self.ll = ll
            self.hl = hl
            self.lim = lim
            self.merge = merge
            self.bg = bg
            self.k_ref = k_ref
            self.inp_scale = inp_scale
            
        def calc(self, inp, trans_dict, use_saved=False):    
            noise_matrix = np.random.normal(self.mean, self.std, inp.shape)
            noise_matrix[noise_matrix<self.ll] = self.ll
            noise_matrix[noise_matrix>self.hl] = self.hl
            noise_matrix = noise_matrix*self.level
            
            #Add some random bg from external array
            if self.bg is not None:
                noise_matrix = noise_matrix + np.array([Image_generator.get_random_bg_from_set(self.bg, inp.shape[2], inp.shape[1]) for i in inp])
            
            if self.inp_scale is not None:
                inp = np.clip(self.inp_scale*inp, 0, 255)
            
            #Make input image brightness as a function of background's brightness. Eg paper and pencil have koeff ~1.35
            if self.k_ref is not None:
                noise_mean = (255-((255-np.mean(noise_matrix, axis=(1,2)).reshape(-1,1,1))/self.k_ref))/255
                if self.inp_scale is not None:
                    inp = np.clip(np.maximum(noise_mean*inp, inp), 0, 255)
                else:
                    inp = np.clip(noise_mean*inp, 0, 255)
                    
            #Merge noise bg and input array
            if self.merge=="full":
                out = noise_matrix + inp
            elif self.merge=="mask":
                noise_matrix[inp>0] = 0
                out = noise_matrix + inp
            elif self.merge=="max":
                out = np.maximum(noise_matrix, inp)
                
            if self.lim == "max":
                out[out>self.level] = self.level
            elif self.lim == "norm":
                out=(out/out.max())*self.level
                
            self.trans_dict["noise"] = (self.mean, self.std, self.ll, self.hl, self.lim)
            trans_dict.update(self.trans_dict)
            return out.astype(int), trans_dict

    ###Preprocess layer: shuffle
    class Mod_shuffle(Mod):
        def __init__(self, shuffle=True, target="all"):
            super().__init__(target=target) 
            self.shuffle = shuffle
        def calc(self, inp, trans_dict, use_saved=False):
            if self.shuffle:
                if not use_saved:
                    self.random = np.random.permutation(len(inp))
                self.trans_dict["shuffle"] = self.shuffle
                return inp[self.random], trans_dict 
            else:
                return inp[self.random], trans_dict 

    ###Preprocess layer: deleting empty images by average level
    class Mod_delete_empty_images(Mod):
        def __init__(self, limit=0, target="all"):
            super().__init__(target=target) 
            self.limit = limit
        def calc(self, inp, trans_dict, use_saved=False):

            if not use_saved:
                self.indice = inp.mean(axis=tuple(range(1,inp.ndim)))>self.limit 

            self.trans_dict["delete_empty_images"] = True
            trans_dict.update(self.trans_dict)

            return inp[self.indice], trans_dict 

    ###Preprocess layer: One hot vector transformation 
    class Mod_one_hot(Mod):
        def __init__(self, list_of_codes, rgb_mode = False, target="y"):
            super().__init__(target=target) 
            self.list_of_codes = list_of_codes
            self.rgb_mode = rgb_mode

        def calc(self, inp, trans_dict, use_saved=False):
            self.trans_dict["one_hot"] = True
            trans_dict.update(self.trans_dict)
            y_colors = np.array([])
            if self.rgb_mode:
                #Using OpenCV colors order: Blue-Green-Red
                y_colors = (inp[:,:,:,2]*65536+inp[:,:,:,1]*256+inp[:,:,:,0])[...,None] 
            else:
                if len(inp.shape) < 4:
                    y_colors = inp.reshape(-1,inp.shape[-2],inp.shape[-1],1)
                else:
                    y_colors = inp
            out = []
            for c in self.list_of_codes:
                out.append((y_colors==c).astype(int)) 
            return np.concatenate(out, axis=3), trans_dict

    ###Preprocess layer: Merge void zones to one specific layer. Default value 
    class Mod_merge_void(Mod):
        def __init__(self, channel_num = 0, target="y"):
            super().__init__(target=target) 
            self.channel_num = channel_num

        def calc(self, inp, trans_dict, use_saved=False):
            self.trans_dict["merge_void"] = self.channel_num
            trans_dict.update(self.trans_dict)
            #Add void zones (without any mapped classes) to specific channel
            inp[...,self.channel_num] = np.equal(np.sum(inp, axis=-1), 0).astype(inp.dtype) + inp[...,self.channel_num]
            return inp, trans_dict

    ###Making RBG y pics from one hot coded multichannel data
    def one_hot_to_rgb(inp, list_of_codes_colors):
        #Using OpenCV colors order: Blue-Green-Red
        #Help function to get BGR codes from DEC
        list_of_colors = np.array([[i - 65536*(i//65536) - 256*((i - 65536*(i//65536))//256), (i - 65536*(i//65536))//256, i//65536] for i in list_of_codes_colors]).astype(np.uint8)
        res_r = np.argmax(inp, axis=3)
        res_g = np.copy(res_r)
        res_b = np.copy(res_r)

        for c in range(inp.shape[-1]):
            res_r[res_r == c] =  list_of_colors[c,2] #2-channel is RED
            res_g[res_g == c] =  list_of_colors[c,1] #1-channel is GREEN
            res_b[res_b == c] =  list_of_colors[c,0] #0-channel is BLUE

        #Using OpenCV colors order: Blue-Green-Red
        return np.concatenate([res_b[...,None],res_g[...,None],res_r[...,None]], axis=3)

    #Shuffle both arrays X and Y together and outputs them reshuffled
    def shuffle_x_y(self, inp_x, inp_y):   
        assert len(inp_x) == len(inp_y)
        p = np.random.permutation(len(inp_x))
        return inp_x[p], inp_y[p]
    
    #Saves data to file
    def save_data(inp_x, inp_y, inp_extra = None, save_dir = "Data/saved_data", name = "exported_data"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "%s.npy"%(name)), 'wb') as f:
            np.save(f, inp_x)
            np.save(f, inp_y)
            if inp_extra is not None:
                np.save(f, inp_extra)
            
    #Load data from
    def load_data(save_dir = "Data/saved_data", name = "exported_data", path=None, extension='.npy'):
        path = path if path is not None else os.path.join(save_dir, "%s%s"%(name,extension))
        with open(path, 'rb') as f:
            out_x = np.load(f)
            out_y = np.load(f)
            out_extra = None
            try:
                out_extra = np.load(f)
            except ValueError:
                pass
        if out_extra is not None:
            return out_x, out_y, out_extra
        else:
            return out_x, out_y
    
    def rgb_to_gray(inp, rgb_weights = [0.2989, 0.5870, 0.1140], invert = False):
        return 255-np.dot(inp[...,:3], rgb_weights) if invert else np.dot(inp[...,:3], rgb_weights)
    
    #Help fulction for loading images and resizing
    def open_image(f, size=None, cv_read_mode = cv.IMREAD_UNCHANGED, interp=cv.INTER_AREA):
        img = cv.imread(f, cv_read_mode)
        if size:
            img = cv.resize(img, size, interpolation = interp) 
        return img
    
    ###Load images from folder. With OpenCV reading mode options, name filtering and resizing
    #cv.IMREAD_UNCHANGED If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
    #cv.IMREAD_GRAYSCALE If set, always convert image to the single channel grayscale image (codec internal conversion).
    #cv.IMREAD_COLOR If set, always convert image to the 3 channel BGR color image.
    #cv.IMREAD_ANYDEPTH If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
    #cv.IMREAD_ANYCOLOR If set, the image is read in any possible color format.
    #cv.IMREAD_LOAD_GDAL If set, use the gdal driver for loading the image.
    #cv.IMREAD_REDUCED_COLOR_2 If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
    ##EXAMPLE: file_filter_regex = r'\d*\.png$'
    def load_from_folder(save_dir = "Data/images", file_filter_regex= r'' , size = None, cv_read_mode = cv.IMREAD_UNCHANGED, recursive=False, interp=cv.INTER_AREA):
        list_of_files = [f for f in glob.glob(f"{save_dir}/**", recursive=recursive) if re.search(file_filter_regex, f) and os.path.isfile(f)]         
        out = np.array([ Image_generator.open_image(f, size=size, cv_read_mode=cv_read_mode, interp = interp) for f in list_of_files])
        return out

    ###Load images from folder for x and y simultaniously. Check if correspondig pair-image exists. With OpenCV reading mode options, name filtering and resizing.
    ##TODO: Refactoring
    def load_from_folders_xy(save_dir_x = "Data/images/x", save_dir_y = "Data/images/y", name_list = None, file_filter_regex= r'' , size = None, cv_read_mode = cv.IMREAD_UNCHANGED, recursive=False, interp_x=cv.INTER_AREA, interp_y=cv.INTER_AREA):
        #Make y blob list
        list_of_files_y = {f:None for f in glob.glob(f"{save_dir_y}/**", recursive=recursive) if os.path.isfile(f)}  
        list_of_files_x = {f:re.search(file_filter_regex, f).group(1) for f in glob.glob(f"{save_dir_x}/**", recursive=recursive) if os.path.isfile(f) and re.search(file_filter_regex, f)} 

        if name_list is not None:
            list_of_files_x = { k:v for (k,v) in list_of_files_x.items() if v in name_list}
            print('Totally %s x images found from name list'%(len(list_of_files_x)))
        else:
            print('Totally %s x images found'%(len(list_of_files_x)))

        #Find partner image x<->y
        for path_x, name_x in list_of_files_x.items():
            regex_y = r'/'+name_x+'\.\w+'
            find_result = [s for s in list_of_files_y if re.search(regex_y, s)]
            list_of_files_x[path_x] = find_result[0] if any(find_result) else None
            if any(find_result):
                list_of_files_y[find_result[0]]=1
        
        no_x_partner_list = [k for (k,v) in list_of_files_x.items() if v is None]
        no_y_partner_list = [k for (k,v) in list_of_files_y.items() if v is None]
        
        if any(no_x_partner_list):
            print('WARNING: Following x has no y images:\n\r',no_x_partner_list)
        if any(no_y_partner_list) and name_list is None:
            print('WARNING: Following y has no x images:\n\r',no_y_partner_list)

        #Filter the result dict
        list_of_files_x = {k:v for (k,v) in list_of_files_x.items() if v is not None}
        print('Valid amount of pairs %s '%(len(list_of_files_x)))

        out_x = np.array([ Image_generator.open_image(f, size=size, cv_read_mode=cv_read_mode, interp=interp_x) for f in list_of_files_x.keys()])
        out_y = np.array([ Image_generator.open_image(f, size=size, cv_read_mode=cv_read_mode, interp=interp_y) for f in list_of_files_x.values()])
        return out_x, out_y   
    
    ###Get file names from folder with desired regex mask. e.g. r'/([^/]+)\.json'.
    def get_file_names_in_folder(files_dir = "Data/images/", recursive=True, file_filter_regex = r'/([^/]+)\.json'):
        return [re.search(file_filter_regex, f).group(1) for f in glob.glob(f"{files_dir}/**", recursive=recursive) if os.path.isfile(f) and re.search(file_filter_regex, f)] 

    ###Load images and make labels from file names
    #EXAMPLE: label_regex = r'/(\d+)\.(cs\.)?png'  for ../00001.cs.png
    #Group number for regex search default = 1 (first () entry)
    def load_from_folder_make_lables(save_dir = "Data/images", file_filter_regex = r'', label_regex = r'/(\d+)\.(cs\.)?png', 
                                     size = None, file_filter_group_num = 1, cv_read_mode = cv.IMREAD_UNCHANGED, recursive=False): 
        list_of_files = [f for f in glob.glob(f"{save_dir}/**", recursive=recursive) if re.search(file_filter_regex, f) and os.path.isfile(f)]         
        out_labels = [re.search(label_regex, f).group(file_filter_group_num) if re.search(label_regex, f) else None for f in list_of_files]

        out_x = np.array([ Image_generator.open_image(f, size=size, cv_read_mode=cv_read_mode) for f in list_of_files])
        
        return out_x, out_labels
    
    def get_random_bg_from_set(inp, w, h):
        rand = (np.random.rand(3)*[inp.shape[0], inp.shape[1]-h, inp.shape[2]-w]).astype(int)
        if len(inp.shape)>3:
            return inp[rand[0],rand[1]:(rand[1]+h),rand[2]:(rand[2]+w),:]
        else:
            return inp[rand[0],rand[1]:(rand[1]+h),rand[2]:(rand[2]+w)]
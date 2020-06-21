import os
import numpy as np
import glob
import cv2 as cv
###DO NOT EDIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Original class usage is in "data_set_preparation.ipynb"
###Specian class for image transformation and augmentation
class Image_generator(object):
    def __init__(self, pipeline):
        self.num_classes = None
        self.num_class_samples = None
        self.pipeline = pipeline
        
    def __call__(self, inp, shuffle = False):
        #Execute pipeline
        pl1_x, pl1_y = self.pipeline(inp)
        
        #Set some data
        self.num_classes = self.pipeline.num_classes
        self.num_class_samples = self.pipeline.num_class_samples
        
        #Output x,y arrays as is or reshuffled
        if shuffle:
            pl1_x, pl1_y = self.shuffle_x_y(pl1_x, pl1_y)
        
        return pl1_x, pl1_y 
    
    ###Pipeline base classes
    class Pipeline(object):
        def __init__(self, common_layers, target_layers, special_layers):
            self.common_layers = common_layers
            self.target_layers = target_layers
            self.special_layers = special_layers
            self.num_classes = None
            self.num_class_samples = None
        
    ###Pipeline with labeles outputs
    class Pipeline_labels(Pipeline):
        def __init__(self, common_layers, target_layers, special_layers):
            super().__init__(common_layers, target_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict)
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                out_trans_dict = inp_trans_dict.copy()
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict)
                    
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
        def __init__(self, common_layers, target_layers, special_layers):
            super().__init__(common_layers, target_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict)
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                out_trans_dict = inp_trans_dict.copy()
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict)
                    
                result_x.append(out) 
                result_y.append(out_trans_dict["rotate_tan"])
                del out, out_trans_dict
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = len(np.unique(result_y))
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            return result_x, result_y  

    ###Pipeline with marked rectangles as target
    class Pipeline_rectangles(Pipeline):
        def __init__(self, common_layers, target_layers, special_layers):
            super().__init__(common_layers, target_layers, special_layers)
        def __call__(self, inp):
            inp_common = inp
            inp_trans_dict = {}
            for i in self.common_layers:
                inp_common, inp_trans_dict = i(inp_common, inp_trans_dict)
            
            result_x = []
            result_y = [] 
            
            #generate Y output
            out_mask_common = np.copy(inp_common)
            for i in self.target_layers:
                out_mask_common, _ = i(out_mask_common, inp_trans_dict)
                
            for p in self.special_layers:
                out = np.copy(inp_common)
                out_mask = np.copy(out_mask_common)
                out_trans_dict = inp_trans_dict.copy()
                
                num_classes = 0
                
                for i in p:
                    out, out_trans_dict = i(out, out_trans_dict)
                    if isinstance(i, Image_generator.Mod_linear_transf):
                        out_mask, _ = i(out_mask, out_trans_dict, use_saved = True)
                    if isinstance(i, Image_generator.Mod_shift):
                        out_mask, _ = i(out_mask, out_trans_dict, use_saved = True)
                    if isinstance(i, Image_generator.Mod_rotate):
                        out_mask, _ = i(out_mask, out_trans_dict, use_saved = True)
                    if isinstance(i, Image_generator.Mod_add_padding):
                        out_mask, _ = i(out_mask, out_trans_dict, use_saved = True)
                
                result_x.append(out) 
                result_y.append(out_mask)
                del out, out_trans_dict, out_mask 
            
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = len(inp_common)
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            del inp_common, out_mask_common
            
            return result_x, result_y  
        
    #Base class for all tranformation layers
    class Mod(object):
        def __init__(self):
            self.trans_dict = {}
    
    #Combine input digits to one long image
    class Mod_combinator(Mod):
        def __init__(self, num_digits, delete_side_margin = 0):
            super().__init__()
            self.delete_side_margin = delete_side_margin
            self.num_digits = num_digits
        def __call__(self, inp, trans_dict):
            #expect input matrix with shape (-1, num_samples, n, m)
            #where n*m - image
            shape = inp.shape[2]
            self.trans_dict["num_combined"] = 4
            trans_dict.update(self.trans_dict)
            return np.concatenate([np.delete(inp[:,i,:,:], list(((k,shape-1-k) for k in range(self.delete_side_margin))), 2) for i in range(self.num_digits)], axis=2), trans_dict
    
    #Add padding left, right, top, bottom
    class Mod_add_padding(Mod):
        def __init__(self, pl=2, pr=2, pt=2, pb=2):
            super().__init__()
            self.padd = (pl,pr,pt,pb)
        def __call__(self, inp, trans_dict, use_saved=False):
            #expect input matrix with shape (-1, a, b)
            inp_shape = list(inp.shape)
            
            inp_shape_l = (inp_shape[0],inp_shape[1], self.padd[0])
            inp_shape_r = (inp_shape[0],inp_shape[1], self.padd[1])
            inp_shape_t = (inp_shape[0],self.padd[2], inp_shape[2]+self.padd[0]+self.padd[1])
            inp_shape_b = (inp_shape[0],self.padd[3], inp_shape[2]+self.padd[0]+self.padd[1])
            self.trans_dict["extra_padding"] = self.padd
            trans_dict.update(self.trans_dict)
            return np.concatenate([np.zeros(inp_shape_t), np.concatenate([np.zeros(inp_shape_l), inp, np.zeros(inp_shape_r)], axis=2), np.zeros(inp_shape_b)], axis=1), trans_dict
    
    #Perspective transformation
    class Mod_linear_transf(Mod):
        ###l,r,t,b percentage of shrinking in corrersponding side (left, right, top, bottom)
        def __init__(self, l=0.0, r=0.0, t = 0.0, b = 0.0, rand = False):
            super().__init__()
            self.inp_sc = (l/2,r/2,t/2,b/2)
            self.rand = rand
        def __call__(self, inp, trans_dict, use_saved=False):
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
        def __init__(self, angle=20.0, rand = False):
            super().__init__()
            self.in_angle = angle
            self.rand = rand
        def __call__(self, inp, trans_dict, use_saved=False):
            w,h = inp.shape[2],inp.shape[1]
            image_center = (w/2,h/2)
            
            if not use_saved:
                self.saved_angle = [self.in_angle if not self.rand else self.in_angle * np.random.rand(1) for i in inp]
                self.shift_matrix = [cv.getRotationMatrix2D(image_center, int(self.saved_angle[num]), 1.0) for num,i in enumerate(inp)]
            
            self.trans_dict["rotate_tan"] = np.tan(2*pi*np.array(self.saved_angle)/360)
            trans_dict.update(self.trans_dict)
            return np.array([cv.warpAffine(i, self.shift_matrix[num], (w,h), flags=cv.INTER_LINEAR) for num,i in enumerate(inp)]), trans_dict
        
    #Shift
    class Mod_shift(Mod):
        ###sh_h, sh_v
        def __init__(self, sh_h=20, sh_v=10, rand = False):
            super().__init__()
            self.sh_h = sh_h
            self.sh_v = sh_v
            self.rand = rand
        def __call__(self, inp, trans_dict, use_saved=False):
            w,h = inp.shape[2],inp.shape[1]
            image_center = (w/2,h/2)
            self.trans_dict["shift"] = (self.sh_h, self.sh_v, self.rand)
            trans_dict.update(self.trans_dict)
            
            if not use_saved:
                self.shift_matrix = np.float32([[[1,0, self.sh_h * (2*np.random.rand(1)-1) if self.rand else self.sh_h],
                                            [0,1, self.sh_v * (2*np.random.rand(1)-1) if self.rand else self.sh_v]] for i in inp])
            return np.array(
                [cv.warpAffine(i, self.shift_matrix[num] ,(w,h),flags=cv.INTER_LINEAR) for num, i in enumerate(inp)]
            ), trans_dict

    #Create recangle
    class Mod_add_rectangle(Mod):
        def __init__(self, t=5.0, l=5.0, w=50, h=30, rand = False):
            super().__init__()
            self.t = t
            self.l = l
            self.w = w
            self.h = h
            self.rand = rand
            
        def __call__(self, inp, trans_dict):    
            yy, xx = np.mgrid[:inp.shape[1], :inp.shape[2]]
            xx = np.repeat(xx[None,...], [inp.shape[0]], axis=0)
            yy = np.repeat(yy[None,...], [inp.shape[0]], axis=0)
            xx[xx<self.l]=0
            xx[xx>(self.l + self.w)]=0
            yy[yy<self.t]=0
            yy[yy>(self.t + self.h)]=0
            mask=xx*yy
            inp[mask>0] = 255
            self.trans_dict["rectangle"] = (self.t, self.l, self.w, self.h)
            trans_dict.update(self.trans_dict)
            return inp, trans_dict 
        
    #Cut random circle in image
    class Mod_round_cut(Mod):
        def __init__(self, r = 10.0, rand_r = False):
            super().__init__()
            self.rand_r = rand_r
            self.r = r
            
        def __call__(self, inp, trans_dict):    
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
        def __init__(self, level=255, std=0.15, mean=0.0, ll=0.0, hl=0.3, lim="max", merge="full", bg = None):  
            super().__init__()
            self.level = level
            self.std = std
            self.mean = mean
            self.ll = ll
            self.hl = hl
            self.lim = lim
            self.merge = merge
            self.bg = bg
            
        def __call__(self, inp, trans_dict):    
            noise_matrix = np.random.normal(self.mean, self.std, inp.shape)
            noise_matrix[noise_matrix<self.ll] = self.ll
            noise_matrix[noise_matrix>self.hl] = self.hl
            noise_matrix = noise_matrix*self.level
            
            #Add some random bg from external array
            if self.bg is not None:
                noise_matrix = noise_matrix + np.array([Image_generator.get_random_bg_from_set(self.bg, inp.shape[2], inp.shape[1]) for i in inp])
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
        
    #Shuffle both arrays X and Y together and outputs them reshuffled
    def shuffle_x_y(self, inp_x, inp_y):   
        combo = np.c_[inp_x.reshape(len(inp_x), -1), inp_y.reshape(len(inp_y), -1)]
        np.random.shuffle(combo)
        x_out = combo[:, :inp_x.size//len(inp_x)].reshape(inp_x.shape)
        y_out = combo[:, inp_x.size//len(inp_x):].reshape(inp_y.shape)
        return x_out, y_out
    
    #Saves data to file
    def save_data(inp_x, inp_y, save_dir = "Data/saved_data", name = "exported_data"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "%s.npy"%(name)), 'wb') as f:
            np.save(f, inp_x)
            np.save(f, inp_y)
            
    #Load data from
    def load_data(save_dir = "Data/saved_data", name = "exported_data"):
        with open(os.path.join(save_dir, "%s.npy"%(name)), 'rb') as f:
            out_x = np.load(f)
            out_y = np.load(f)
        return out_x, out_y
    
    def rgb_to_gray(inp, rgb_weights = [0.2989, 0.5870, 0.1140], invert = False):
        return 255-np.dot(inp[...,:3], rgb_weights) if invert else np.dot(inp[...,:3], rgb_weights)
    
    def load_from_folder(save_dir = "Data/images", ext="", size = None):
        def open_image(f, size=None):
            img = cv.imread(f, cv.IMREAD_UNCHANGED)
            if size:
                img = cv.resize(img, size, interpolation = cv.INTER_AREA) 
            return img
        
        list_of_files = [f for f in glob.glob(f"{save_dir}/*{ext}")]             
        out = np.array([ open_image(f,size) for f in list_of_files if os.path.isfile(f)])
        return out
    
    def get_random_bg_from_set(inp, w, h):
        rand = (np.random.rand(3)*[inp.shape[0], inp.shape[1]-h, inp.shape[2]-w]).astype(int)
        if len(inp.shape)>3:
            return inp[rand[0],rand[1]:(rand[1]+h),rand[2]:(rand[2]+w),:]
        else:
            return inp[rand[0],rand[1]:(rand[1]+h),rand[2]:(rand[2]+w)]


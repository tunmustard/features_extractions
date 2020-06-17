import os
import numpy as np
###DO NOT EDIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Original class usage is in "data_set_preparation.ipynb"
###Specian class for image transformation and augmentation
class Image_generator(object):
    def __init__(self):
        
        #Common transformation layers
        ###Should be overwritten
        
        #Special transformation layers
        ###Should be overwritten
        
        #Create pipeline structure
        self.pipelene_1 = Image_generator.Pipeline(
            common_layers = [],
            special_layers = [
                []
            ]
        )
        
        self.num_classes = None
        self.num_class_samples = None
        
    def __call__(self, inp, shuffle = False):      
        #Execute pipeline
        pl1_x, pl1_y = self.pipelene_1(inp)
        
        #Set some data
        self.num_classes = self.pipelene_1.num_classes
        self.num_class_samples = self.pipelene_1.num_class_samples
        
        #Output x,y arrays as is or reshuffled
        
        if shuffle:
            pl1_x, pl1_y = self.shuffle_x_y(pl1_x, pl1_y)
        
        return pl1_x, pl1_y 
        
    ###Pipeline class
    class Pipeline(object):
        def __init__(self, common_layers, special_layers):
            self.common_layers = common_layers
            self.special_layers = special_layers
            self.num_classes = None
            self.num_class_samples = None
        def __call__(self, inp):
            inp_common = inp
            for i in self.common_layers:
                inp_common = i(inp_common)
            
            result_x = []
            result_y = [] 

            for p in self.special_layers:
                out = np.copy(inp_common)
                for i in p:
                    out = i(out)
                result_x.append(out) 
                result_y.append(np.array([i for i in range(inp_common.shape[0])]))
   
            result_x = np.concatenate(result_x,axis=0)
            result_y = np.concatenate(result_y,axis=0)
            
            self.num_classes = result_y.max()+1
            self.num_class_samples = int(result_x.shape[0]/(self.num_classes))
            
            return result_x, result_y
        
    #Combine input digits to one long image
    class Mod_combinator(object):
        def __init__(self, num_digits, delete_side_margin = 0):
            self.delete_side_margin = delete_side_margin
            self.num_digits = num_digits
        def __call__(self, inp):
            #expect input matrix with shape (-1, num_samples, n, m)
            #where n*m - image
            shape = inp.shape[2]
            return np.concatenate([np.delete(inp[:,i,:,:], list(((k,shape-1-k) for k in range(self.delete_side_margin))), 2) for i in range(self.num_digits)], axis=2)
    
    #Add padding left, right, top, bottom
    class Mod_add_padding(object):
        def __init__(self, pl=2, pr=2, pt=2, pb=2):
            self.padd = (pl,pr,pt,pb)
        def __call__(self, inp):
            #expect input matrix with shape (-1, a, b)
            inp_shape = list(inp.shape)
            inp_shape_l = (inp_shape[0],inp_shape[1], self.padd[0])
            inp_shape_r = (inp_shape[0],inp_shape[1], self.padd[1])
            inp_shape_t = (inp_shape[0],self.padd[2], inp_shape[2]+self.padd[0]+self.padd[1])
            inp_shape_b = (inp_shape[0],self.padd[3], inp_shape[2]+self.padd[0]+self.padd[1])
            return np.concatenate([np.zeros(inp_shape_t), np.concatenate([np.zeros(inp_shape_l), inp, np.zeros(inp_shape_r)], axis=2), np.zeros(inp_shape_b)], axis=1)
    
    #Perspective transformation
    class Mod_linear_transf(object):
        ###l,r,t,b percentage of shrinking in corrersponding side (left, right, top, bottom)
        def __init__(self, l=0.0, r=0.0, t = 0.0, b = 0.0, rand = False):
            self.inp_sc = (l/2,r/2,t/2,b/2)
            self.rand = rand
        def __call__(self, inp):
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
            return np.array([cv.warpPerspective(i,M,(w,h)) for i in inp])
        
    #Cut random circle in image
    class Mod_round_cut(object):
        def __init__(self, r = 10.0, rand_r = False):
            self.rand_r = rand_r
            self.r = r
            
        def __call__(self, inp):    
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
            return inp*mask
        
    #Add noise
    class Mod_add_noise(object):
        def __init__(self, level=255, std=0.15, mean=0.0, ll=0.0, hl=0.3, lim="max"):  #limiting "max" or "norm"
            self.level = level
            self.std = std
            self.mean = mean
            self.ll = ll
            self.hl = hl
            self.lim = lim
            
        def __call__(self, inp):    
            noise_matrix = np.random.normal(self.mean, self.std, inp.shape)
            noise_matrix[noise_matrix<self.ll] = self.ll
            noise_matrix[noise_matrix>self.hl] = self.hl
            noise_matrix = noise_matrix*self.level
            out = noise_matrix + inp
            if self.lim == "max":
                out[out>self.level] = self.level
            else:
                out=(out/out.max())*self.level
            return out.astype(int)
        
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
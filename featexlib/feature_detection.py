###CLASS ORIGIN IS HERE
###Special class for finding similar feature vectors

import numpy as np

class Feature_detection(object):
    def __init__(self, feature_shape, buff_len = 5, center_norm_lim=0.5, center_dist_norm_lim=0.5, likeness_lim = 0.5):
        self.id_list = np.array([])
        self.ft_shape = feature_shape
        self.buff_len = buff_len
        self.ft_buffer = np.zeros( (buff_len, ) + feature_shape , dtype='float32') 
        self.center_norm_lim = center_norm_lim
        self.center_dist_norm_lim = center_dist_norm_lim
        self.likeness_lim = likeness_lim

    def add_to_buffer(self, inp):
        self.ft_buffer = np.vstack([np.delete(self.ft_buffer, 0, 0),inp])

        
    def get_vector_from_buffer(self):
        
        center = np.mean(self.ft_buffer, axis=0)
        center_norm = np.linalg.norm(center)
        center_dist = self.ft_buffer-center
        center_dist_norm = np.linalg.norm(center_dist, axis=1)
        print("center_dist_norm.max()",center_dist_norm.max())
        print("center_norm",center_norm)
        if center_norm>self.center_norm_lim and center_dist_norm.max()<self.center_dist_norm_lim:
            return center
        return None
        
    def get_id(self, inp):  
        inp = inp[None,...]
        if len(self.id_list) > 0:
            likeness_dist_norm_arr = np.linalg.norm(self.id_list - inp, axis=1)
            likeness_dist_norm_min = likeness_dist_norm_arr.min()
            likeness_indice = likeness_dist_norm_arr.argmin()
            print("likeness_dist_norm_min",likeness_dist_norm_min)
            if likeness_dist_norm_min < self.likeness_lim:
                return likeness_indice+1
            else:
                self.id_list = np.vstack([self.id_list,inp])
                return len(self.id_list)
        else:
            self.id_list = inp
            return len(self.id_list)
        
    def __call__(self, inp):
        
        if inp is None:
            inp = np.zeros(self.ft_shape)
        inp = np.array(inp)
        
        self.add_to_buffer(inp)
        
        vector = self.get_vector_from_buffer();
        
        if vector is not None:
            inp_id = self.get_id(inp)
        else:
            inp_id = None
        
        return inp_id
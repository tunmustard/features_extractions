import os
import matplotlib.pyplot as plt
import numpy as np
###This is class origin. Please use this class.
###Helpful class for debuging. Shows images, convert image batches as matrix to rgb, rescales...
class Debug():
    class Image():
        def __init__(self,image,label=""):
            self.image = image
            self.label = label
            
    def show_images_list(image_list, labels = [], cmap = 'gray', col_number = 10, height = 2, save_name=None, save_dir = "Data/saved_images", vmin=None, vmax=None):
        row = -(-len(image_list)//col_number) 
        fig = plt.figure(figsize=(15, row*height))
        count = 1
        
        if save_name:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        for i in range(len(image_list)):
            a = fig.add_subplot(row, col_number, count)
            plt.axis('off')
            if isinstance(i, Debug.Image):
                if image_list[i].label:
                    a.set_title(image_list[i].label)
                plt.imshow(image_list[i].image, cmap = cmap, vmin=vmin, vmax=vmax) 
            else:
                if any(labels):
                    a.set_title(labels[i])
                out_img = plt.imshow(image_list[i], cmap = cmap) 
            if save_name:
                plt.imsave(os.path.join(save_dir, "%s_%s.png"%(save_name,i)), image_list[i], cmap = cmap)
            count=count+1
            
    def show_image(img, cmap = 'gray', no_axis = False, vmin=None, vmax=None):
        if no_axis:
            plt.axis('off')
        plt.imshow(img, cmap = cmap, vmin=vmin, vmax=vmax) 
        
    def scale_to_1(x):
        return (x/255.0)
    
    def scale_to_255(x):
        return (x*255.0)
        
    def make_rgb_from_gray(img, mode="r"):
        img = img.reshape(-1,28,28,1)
        out_shape = list(img.shape)
        out_shape[-1] = 2
        out = np.concatenate((img, np.zeros(out_shape)), axis=3)
        if mode == "r":
            return out
        if mode == "g":
            return out[:,:,:,[1,0,2]]
        if mode == "b":
            return out[:,:,:,[1,2,0]]
    #Load data from
    def load_data(save_dir = "Data/saved_data", name = "exported_data"):
        with open(os.path.join(save_dir, "%s.npy"%(name)), 'rb') as f:
            out_x = np.load(f)
            out_y = np.load(f)
        return out_x, out_y
import numpy as np
import cv2 as cv
import tensorflow as tf
from utils import vstack, normalize
import glob
class Loader():
    def __init__(self, params):
        self.img_path = params.img_path
        self.attr_path = params.attr_path
        self.attributes =  self.prepare_attributes(params)
        self.n_images = params.n_images

        # Séparation des datasets 
        self.train_indices = int(self.n_images*0.8)
        self.val_indices = self.train_indices + int(self.n_images*0.1)
        self.test_indices = self.n_images

        #Data augmentaiton
        self.h_flip = params.h_flip
        self.v_flip = params.v_flip

        self.data_loaded_in_ram = params.load_in_ram
        if self.data_loaded_in_ram:
            self.images = self.load_images_in_ram(params)
        
    
    def load_images_in_ram(self, params):
        """
        Two mode of loading : 
            eitwher we load from a directory and we process images directly, (loading, rescaling, normalizing)
            or we load from numpy load where data have been preprocessed
        """
        indices = range(1, params.n_images + 1)
        self.images  =  np.array([])
        
        print("Loading images in memory")
        if params.loading_mode == "preprocessed":
            self.images = np.load(self.img_path)['arr_0']
        else:
            for i in indices:
                im = self.load_image(i, normal= True)
                im = np.expand_dims(im, 0)
                self.images = vstack(self.images, im)
                if i % 100 == 0:
                    print(f"{i} / {params.n_images}")
        print("Finished loading")


    def load_random_batch(self, ind_min, ind_max, bs):
        indices = np.random.randint(ind_min, ind_max, bs)
        return self.load_batch(indices)

    def load_batch_sequentially(self,ind_min, ind_max):
        indices = range(ind_min, ind_max) 
        return self.load_batch(indices)

    def load_batch(self, indices):
        """
        indices: tableau contenant les idnices des images a charger
        """
        batch_x, batch_y = np.array([]), np.array([])

        for i in indices:
            if not self.data_loaded_in_ram:
                im = self.load_image(i)
            else:
                im = self.images[i-1]

            im = np.expand_dims(im, 0)
            batch_x = vstack(batch_x, im)
            batch_y = vstack(batch_y, self.attributes[i-1]) #i - 1 important 
            
        return tf.constant(batch_x), tf.constant(batch_y)   

    def load_image(self,i, normal = True):
        im = cv.imread(self.img_path+"/%06i.jpg" %i)
        shp = im.shape


        if shp[0] != 256 or shp[1] != 256:
            im = cv.resize(im, (256,256), interpolation=cv.INTER_LANCZOS4)

        if normal:
            im = normalize(im)
        
        if self.h_flip and np.random.rand() <= 0.5:
            im = np.flip(im, axis = 1)

        if self.v_flip and np.random.rand() <= 0.5:
            im = np.flip(im, axis = 0)

        return im

    def prepare_attributes(self, params):
        #params.attr est un string : 
        attributes = np.load(self.attr_path, allow_pickle=True)['arr_0'].item()
        if not isinstance(params.attr, (list, np.ndarray)):
            if params.attr == '*':
                params.attr = list(attributes.keys())

            else:
                # np.unique evite que l'utilisateur entre 2 fois le meme attribut
                params.attr = np.unique(params.attr.replace(' ', '').split(','))
                check = np.isin(params.attr, list(attributes.keys()))
                if not check.all():
                    raise ValueError(f"Les attribut {params.attr[np.where(check == False)]} ne sont pas pris en compte")
        

        params.n_attr = len(params.attr)
        attr = None
        for p in params.attr:
            for i in range(2):
                formated_attr = np.reshape(attributes[p] == i, (len(attributes[p]), 1))
                if attr is None:
                    attr = np.array(formated_attr.astype(np.float32))
                else:
                    attr = np.append(attr, formated_attr,axis = 1)
        return attr

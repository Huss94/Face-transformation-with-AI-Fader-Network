import numpy as np
import cv2 as cv
import tensorflow as tf
from utils import vstack, normalize
class Loader():
    def __init__(self, params, train_ind, val_ind):
        self.img_path = params.img_path
        self.attr_path = params.attr_path
        self.attributes =  self.prepare_attributes(params)
        self.train_ind = train_ind
        self.val_ind = val_ind
        
    
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
            im = cv.imread(self.img_path+"/%06i.jpg" %i)
            im = cv.resize(im, (256,256), interpolation=cv.INTER_LANCZOS4)
            im = normalize(im)
            im = np.expand_dims(im, 0)
            batch_x = vstack(batch_x, im)
            batch_y = vstack(batch_y, self.attributes[i])
        
        return tf.constant(batch_x), tf.constant(batch_y)   


    def prepare_attributes(self, params):
        considered_attr = params.attr.replace(' ', '').split(',')
        params.n_attr = len(considered_attr)
        attributes = np.load(self.attr_path, allow_pickle=True)['arr_0'].item()
        attr = None
        for p in considered_attr:
            for i in range(2):
                formated_attr = np.reshape(attributes[p] == i, (len(attributes[p]), 1))
                if attr is None:
                    attr = np.array(formated_attr.astype(np.float32))
                else:
                    attr = np.append(attr, formated_attr,axis = 1)
        return attr

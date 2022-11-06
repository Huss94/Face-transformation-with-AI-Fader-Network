import numpy as np
import cv2 as cv
import tensorflow as tf
from utils import vstack, normalize
class Loader():
    def __init__(self, attributes_path, images_path, considered_attributes, train_ind, val_ind):
        self.attributes =  self.prepare_attributes(attributes_path, considered_attributes)
        self.images_path = images_path
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
            im = cv.imread(self.images_path+"/%06i.jpg" %i)
            im = cv.resize(im, (256,256), interpolation=cv.INTER_LANCZOS4)
            im = normalize(im)
            im = np.expand_dims(im, 0)
            batch_x = vstack(batch_x, im)
            batch_y = vstack(batch_y, self.attributes[i])
        
        return tf.constant(batch_x), tf.constant(batch_y)   


    def prepare_attributes(self, attributes_path,param_attr):
        attributes = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item()
        attr = None
        for p in param_attr:
            for i in range(2):
                formated_attr = np.reshape(attributes[p] == i, (len(attributes[p]), 1))
                if attr is None:
                    attr = np.array(formated_attr.astype(np.float32))
                else:
                    attr = np.append(attr, formated_attr,axis = 1)
        return attr

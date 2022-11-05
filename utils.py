import numpy as np
import cv2 as cv

def prepare_attributes(attributes,param_attr):
    attr = None
    for p in param_attr:
        for i in range(2):
            formated_attr = np.reshape(attributes[p] == i, (len(attributes[p]), 1))
            if attr is None:
                attr = np.array(formated_attr.astype(np.float32))
            else:
                attr = np.append(attr, formated_attr,axis = 1)
    return attr



def vstack(array1, array2):
    try:
        new_array = np.vstack((array1,array2))
    except ValueError:
        if len(array1) == 0:
            return array2
        elif len(array2) == 0:
            return array1

    return new_array


def load_batch(ind_min, ind_max, bs, attributes):
    indices = np.random.randint(ind_min, ind_max, bs)
    imgs = [] 
    batch_x, batch_y = np.array([]), np.array([])

    for i in indices:
        im = cv.imread("data/img_align_celeba/%06i.jpg" %i)
        im = cv.resize(im, (256,256), interpolation=cv.INTER_LANCZOS4)
        im = normalize(im)
        im = np.expand_dims(im, 0)
        batch_x = vstack(batch_x, im)
        batch_y = vstack(batch_y, attributes[i])
    
    return batch_x, batch_y


def normalize(image):    
    # La normalisation utilisée par les auteurs de la technique est différente, 
    # return image/127.5 -1
    return image/255.0


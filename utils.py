import numpy as np
import cv2 as cv
import tensorflow as tf


def vstack(array1, array2):
    try:
        new_array = np.vstack((array1,array2))
    except ValueError:
        if len(array1) == 0:
            return array2
        elif len(array2) == 0:
            return array1

    return new_array





def normalize(image):    
    # Normalization entre -1 et 1 
    return image/127.5 -1



import numpy as np
import cv2 as cv
import tensorflow as tf
import os 

AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]


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

def save_model(model, name, folder_name= 'models'):
    if not  os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    model.save(folder_name + '/' + name)


#Cette fonction est grandement inspirée de la fonction utilisée par les developpeurs du fader network
#Elle ne sert qu'a vérifié si les attributs entré par les utilisateurs sont bons
def check_attr(params):
    """
    Check attributes validy.
    """
    if params.attr == '*':
        params.attr = attr_flag(','.join(AVAILABLE_ATTR))
    else:
        assert all(name in AVAILABLE_ATTR and n_cat >= 2 for name, n_cat in params.attr)
    params.n_attr = sum([n_cat for _, n_cat in params.attr])

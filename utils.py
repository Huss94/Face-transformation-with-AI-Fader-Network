import numpy as np
import tensorflow as tf
import os 
import pickle
from model import Classifier

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

def save_model(model, name, folder_name= 'models', pickle_save = False):
    if not  os.path.isdir(folder_name):
        os.mkdir(folder_name)

    print("Enrigistrement du model")
    if pickle_save: 
        filehandler = open(folder_name + '/' + name + '.pkl', 'wb')
        pickle.dump(model, filehandler)
        filehandler.close()
    else:
        model.save(folder_name + '/' + name)


def save_model_weights(model, name, folder_name ='models'):
    print("Sauvegarde des poids du reseau")
    model.save_weights(folder_name + '/' + name + '_weights')
    np.save(folder_name + '/' + name + '_params', model.params)

def load_classifier(path, params = '_params.npy', weights ='_weights'):
    c_params = np.load(path + params, allow_pickle = True).item()
    C = Classifier(c_params)
    C.load_weights(path + weights)
    return C
import numpy as np
import tensorflow as tf
import os 
import pickle
from model import Classifier, Fader, AutoEncoder
import glob

def vstack(array1, array2):
    try:
        new_array = np.vstack((array1,array2))
    except ValueError:
        if len(array1) == 0:
            return array2
        elif len(array2) == 0:
            return array1

    return new_array

def hstack(array1, array2):
    try:
        new_array = np.hstack((array1,array2))
    except ValueError:
        if len(array1) == 0:
            return array2
        elif len(array2) == 0:
            return array1

    return new_array



def normalize(image):    
    # Normalization entre -1 et 1 
    return image/127.5 -1

def denormalize(image):
    im = np.array(image)
    im = 127.5*(im + 1)
    return np.uint8(im)

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


def save_model_weights(model, name, history, folder_name ='models'):
    """Save the network weights

    Args:
        model (keras.model): model to save
        name (str): name of the folder we save the model in
        history (dict): training history 
        folder_name (str, optional): folder in which we save the models. Defaults to 'models'.
    """
    print("Sauvegarde des poids du reseau")
    model.save_weights(folder_name + '/' + name + '/' + 'weights')
    np.save(folder_name + '/' + name + '/' + 'params', model.params)
    np.save(folder_name + '/' + name + '/history.npy', history)

def load_model(path, model_type, params_name = 'params.npy', weights_name ='weights' , train = False):
    """Load a model

    Args:
        path (str): directory ot the model to load
        model_type (str): 'c' for a classifier, 'f' for a fader,  'ae' for an AutoEncoder
        params_name (str, optional): parameters of the model name. Defaults to 'params.npy'.
        weights_name (str, optional): weights of the models. Defaults to 'weights'.
        train (bool, optional): charge the model train mode or not. Defaults to False.

    Raises:
        ValueError: If model type is of not known type

    Returns:
        keras.model : the loaded model
    """

    params = np.load(path + '/'+ params_name, allow_pickle = True).item()
    if model_type == 'c':
        model = Classifier(params)
    elif model_type =='f':
        model = Fader(params)
    elif model_type =='ae':
        model = AutoEncoder(params)
    else: 
        raise ValueError(f"invalid model_type = {model_type}, possible value are 'c', 'f' or 'ae'")

    model.load_weights(path + '/' + weights_name)
    model.trainable = train

    return model


def load_history(path : str, name = "history.npy"):
    """Load history from path of the folder. Return None if nothing was find

    Args:
        path (str): Path of the foler
        name (str, optional): Name of the history file. Defaults to "history.npy".

    Returns:
        dict: history 
    """

    for i in range(2):
        if os.path.isfile(path +   '/'+ name):
            h = np.load(path+ '/' + name , allow_pickle = True).item()
            return h 
        else:
            if '/' in path:
                # Dans le cas où l'historique est dans le dossier parent
                index = path[::-1].index('/') 
                path = path[:len(path) - index - 1]
    return None
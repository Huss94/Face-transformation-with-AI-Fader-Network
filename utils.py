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


def save_model_weights(model, name, folder_name ='models', get_optimizers = False):
    print("Sauvegarde des poids du reseau")
    model.save_weights(folder_name + '/' + name + '/' + 'weights')
    np.save(folder_name + '/' + name + '/' + 'params', model.params)

    # if get_optimizers:
    #     for i, opt in enumerate(model.get_optimizers()):
    #         if not os.path.isdir(folder_name + '/' + name + '/' + 'optimizers'):
    #             os.mkdir(folder_name + '/' + name + '/' + 'optimizers')

    #         np.save(folder_name + '/' + name + '/' + 'optimizers/' + str(i), opt.get_weights(), allow_pickle=True)

def load_model(path, model_type, params_name = 'params.npy', weights_name ='weights' , train = False):
    """
    Charge un model, uniqnument pour l'inférence ce modèle ne peut pas etre entrainer étant donnée qu'on enrigistre pas le statut des optimizers 
    -----  
    Parameters : 
    path : str, chemin vers le dossier contenant le modèle
    param_name : str, nom du fichier contenant les paramètres du modele
    weights_name : str, nom du fichier contenantl les poids du modèle
    model_type : str, 'c' pour un classifier, 'f' pour un fader, 'ae' pour un autoencoder
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

    model.load_weights(path + '/' +weights_name)
    model.trainable = train
    
    # if restore_optimizers:
    #     opts = []
    #     if not os.path.isdir(path + '/optimizers'): 
    #         raise ValueError("Aucun optimizer trouvé")
    #     opts_path = glob.glob(path + '/optimizers/*')
    #     assert len(opts_path) != 0
    #     for opt in opts_path: 
    #         w  = np.load(opt, allow_pickle=True)
    #         opts.append(w) 
    #     return model, opts
    return model


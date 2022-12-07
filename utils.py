import numpy as np
import tensorflow as tf
import os 
import pickle
from model import Classifier, Fader, AutoEncoder
import glob

from matplotlib.figure import Figure

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

def trace_history(h : dict):
    if h is not None:
        n_epoch = len(h[list(h.keys())[0]])
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)

        for i, attr in enumerate(h):
            if 'loss' in attr and 'classifier' not in attr:
                a.plot(np.arange(0,n_epoch), h[attr], label = attr)
                
        a.legend()
        a.set_title("Tracé des loss à travers les époch")
        a.set_xlabel("epochs")
        a.set_ylabel("Loss")
    
        return f

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


def dic_to_tab(dic, exception = []):
    """
    Transform a dictionnary to a table of tuple 

    Args:
        dic (dict): dictonarry of metrics

    Returns:
        table of tuple (keys, value)
    """
    tab = []
    for k in dic:
        if k not in exception:
            tab.append((k,dic[k]))

    return tab


def fill_tab(tab, desired_size, value):
    diff = desired_size - len(tab)
    assert diff > 0

    diff_tab = [value]*diff
    tab += diff_tab



class Metrics:
    """Class to handle metrics easily 
    """
    def __init__(self, *metrics):
        self.epoch = 0
        self.metrics = metrics
        self.mean_dic = {}

        for i in metrics:
            self.mean_dic[i] = [None]
        
    def __getitem__(self, value : str):
        return self.mean_dic[value]
    
    def add_metrics(self,*metrics):
        for m in metrics:
            self.metrics.append(m)
    
    def update(self,dic):
        """Update values in the dictionnary and compute directly the mean of each metrics

        Args:
            dic (dict): Dictonnary of metrics and current iteration value
        """
        for p in dic:
            if p in self.mean_dic:
                if isinstance(dic[p], tf.Tensor):
                    dic[p] = dic[p].numpy()
                try:
                    current = self.mean_dic[p][self.epoch]
                except IndexError:
                    fill_tab(self.mean_dic[p], self.epoch + 1, None)
                    current = self.mean_dic[p][self.epoch]

                if  current is None or np.isnan(current):
                   self.mean_dic[p][self.epoch] = dic[p]
                else:
                    self.mean_dic[p][self.epoch] =  (current + dic[p])/2
    
    def new_epoch(self):
        self.epoch +=1
        for m in self.metrics : 
            self.mean_dic[m].append(None)

    def load(self,dic):
        self.mean_dic = dic
        best = 0
        for m in self.mean_dic:
            l = len(self.mean_dic[m])
            if l > best:
                best = l
        self.epoch = best - 1
        self.new_epoch()


    
    def load_dic_from_path(self, folder_path, name = "history.npy"):
        self.load(load_history(folder_path, name))

    
    def save(self,path, name):
        np.save(path + "/" + name, self.mean_dic)
        
    def __len__(self):
        return self.epoch



        

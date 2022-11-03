import numpy as np 
import tensorflow as tf
from utils import *
from model import AutoEncoder
import cv2 as cv
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Chargement des données 
    ae = AutoEncoder(4)

    all_data = np.load("data/test.npz")['arr_0']/255
    all_data = all_data[:600]
    attributes = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item()

    #Params attributes
    param_attr = ('Male', 'Smiling')
    attr = prepare_attributes(attributes, param_attr)
    
    train_indices = 500
    val_indices = 600
    attr = attr[:600,:]

    x_train = all_data[:10]
    y_train = attr[:10]
    x_val = all_data[train_indices:val_indices]
    print("----------------------------------")

    enc, newim = ae(x_train, y_train)
    print(newim.shape)
    
    print(newim)
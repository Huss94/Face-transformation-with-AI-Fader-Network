import numpy as np 
import tensorflow as tf
from tensorflow import keras
from utils import *
from model import AutoEncoder, Fader
import cv2 as cv
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Chargement des données 
    ae = AutoEncoder(4)

    all_data = np.load("data/test.npz")['arr_0']/127.5
    all_data = all_data[:600]
    attributes = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item()

    #Params attributes
    param_attr = ('Male', 'Smiling')
    attr = prepare_attributes(attributes, param_attr)
    
    train_indices = 500
    val_indices = 600
    attr = attr[:600,:]

    x_train = tf.constant(all_data[:train_indices])
    y_train = tf.constant(attr[:train_indices])

    x_val = all_data[train_indices:val_indices]
 
    bs = 5
    f = Fader(ae)
    
    f.compile(
        ae_opt= keras.optimizers.Adam(learning_rate=0.0002),
        dis_opt= keras.optimizers.Adam(learning_rate=0.0002),
        ae_loss = keras.losses.MeanSquaredError(),
        run_eagerly=False
    )

    f.fit(x_train,y_train, batch_size = bs, epochs = 50)
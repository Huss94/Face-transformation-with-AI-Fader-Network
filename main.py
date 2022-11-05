import numpy as np 
import tensorflow as tf
from tensorflow import keras
from utils import *
from model import AutoEncoder, Fader
import cv2 as cv
import matplotlib.pyplot as plt
from utils import load_batch
from time import time
from tqdm import tqdm





if __name__ == "__main__":
    # Chargement des données 
    ae = AutoEncoder(4)

    # all_data = np.load("data/test.npz")['arr_0']/127.5
    # all_data = all_data[:600]
    attributes = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item()

    #Params attributes
    param_attr = ('Male', 'Smiling')
    attr = prepare_attributes(attributes, param_attr)
    
    train_indices = 160000
    val_indices = 200000

    # x_train = tf.constant(all_data[:train_indices])
    # y_train = tf.constant(attr[:train_indices])

    # x_val = all_data[train_indices:val_indices]
 
    bs = 32
    epochs = 5
    f = Fader(ae)
    
    f.compile(
        ae_opt= keras.optimizers.Adam(learning_rate=0.0002),
        dis_opt= keras.optimizers.Adam(learning_rate=0.0002),
        ae_loss = keras.losses.MeanSquaredError(),
        run_eagerly= False
    )

    for epoch in range(epochs):
        for step in tqdm(range(0, train_indices//bs, bs)):
            t = time()
            batch_x, batch_y = load_batch(0, train_indices, bs, attr)
            recon_loss, dis_loss = f.custom_train_step((batch_x, batch_y))
            print(step, recon_loss, dis_loss,  "time : ", time() - t)





            # Validation
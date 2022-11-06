import numpy as np 
import tensorflow as tf
from tensorflow import keras
from utils import *
from model import AutoEncoder, Fader
from loader import Loader
import cv2 as cv
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import argparse


#L'utilisation du parser a été inspiré du code impélmenté par les développeur du fader network
parser = argparse.ArgumentParser(description='Train the fader Network')

parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
parser.add_argument("--img_path", type = str, default = "data/img_align_celeba", help= "Path to images")
parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
parser.add_argument("--attr", type = str, default= "Smiling,Male", help= "Considered attributes to train the network with")
parser.add_argument("--n_epoch", type = int, default = 1000, help = "Numbers of epochs")
parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")

params = parser.parse_args()

if __name__ == "__main__":
    train_indices = 162770
    val_indices = train_indices + 19867

    Data = Loader(params, train_indices, val_indices)
    bs = 5

    # A augmenter en fonction de la mémoire disponible dans l'ordinateur
    # On perd un peu de temps a recharger a chaque fois les images de validation alors que se sont toujours les 20000 memes
    # Il faudrait peut etre changer ca si on a accès a plus de RAM et gardre en mémoire le dataset de validation
    eval_bs = 100
    epochs = 6
    f = Fader(params)
    
    f.compile(
        ae_opt= keras.optimizers.Adam(learning_rate=0.0002),
        dis_opt= keras.optimizers.Adam(learning_rate=0.0002),
        ae_loss = keras.losses.MeanSquaredError(),
        run_eagerly= False
    )

    # tf.config.run_functions_eagerly(True)
    history = {}
    history['reconstruction_loss'] = []
    history['discriminator_loss'] = []
    history['dis_accuracy'] = []
    for epoch in range(epochs):

        #Training
        recon_loss_tab = []
        dis_loss_tab = []
        dis_accuracy_tab = []

        for step in range(0, train_indices//bs, bs):
            t = time()
            batch_x, batch_y = Data.load_random_batch(0, train_indices, bs)
            recon_loss, dis_loss,  dis_acc= f.train_step((batch_x, batch_y))

            recon_loss_tab.append(recon_loss)
            dis_loss_tab.append(dis_loss)
            dis_accuracy_tab.append(dis_acc)
            print(step, dis_acc, time()-t)
            
            if step >= bs * 3:
                break


        history['reconstruction_loss'].append(np.mean(recon_loss_tab))
        history['discriminator_loss'].append(np.mean(dis_loss_tab))
        history['dis_accuracy'].append(np.mean(dis_accuracy_tab))

        # Validation
        recon_val_loss = []
        dis_val_loss = []
        dis_val_accuracy = []

        for step in range(train_indices, val_indices, eval_bs):
            t = time()
            batch_x, batch_y = Data.load_batch_sequentially(step, step+eval_bs)
            recon_loss, dis_loss, dis_acc = f.evaluate_on_val((batch_x, batch_y))

            recon_val_loss.append(recon_loss)
            dis_val_loss.append(dis_loss)
            dis_val_accuracy.append(dis_acc)
            print(time() - t)

        history['reconstruction_val_loss'].append(np.mean(recon_val_loss))
        history['discriminator_val_loss'].append(np.mean(dis_val_loss))
        history['dis_val_accuracy'].append(np.mean(dis_val_accuracy))

       

    
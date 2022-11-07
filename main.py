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
parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images. It can be the directory of the image, or the npz file")
parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
parser.add_argument("--attr", type = str, default= "Smiling,Male", help= "Considered attributes to train the network with")
parser.add_argument("--n_epoch", type = int, default = 1000, help = "Numbers of epochs")
parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")
parser.add_argument("--loading_mode", type = str, default = "preprocessed", help = "2 values : 'preprocessed' or 'direct'. from what the data are loaded npz file or direct data")
parser.add_argument("--load_in_ram", type= bool, default = False, help = "Si l'ordinateur n'a pas assez de ram pour charger toutes les données en meme temps, mettre False, le programme chargera seuleemnt les batchs de taille défini (32 par default) puis les déchargera après le calcul effectué") 
parser.add_argument("--resize", type= bool, default = False, help = "Applique le resize a chaque fois qu'une donnée est chargée. Mettre a False si les images on été resized en amont") 
parser.add_argument("--save_path", type= str, default = "models", help = "Indique où enrisitrer le model") 

params = parser.parse_args()

if __name__ == "__main__":
    train_indices = 162770
    val_indices = train_indices + 19867

    Data = Loader(params, train_indices, val_indices)

    #eval_bs est le nombre de fichier a charger d'un coup dans la ram 
    eval_bs = 100

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
    for epoch in range(params.n_epoch):

        #Training
        recon_loss_tab = []
        dis_loss_tab = []
        dis_accuracy_tab = []

        for step in range(0, params.epoch_size, params.batch_size):
            t = time()
            batch_x, batch_y = Data.load_random_batch(1, train_indices, params.batch_size)
            recon_loss, dis_loss,  dis_acc= f.train_step((batch_x, batch_y))

            recon_loss_tab.append(recon_loss)
            dis_loss_tab.append(dis_loss)
            dis_accuracy_tab.append(dis_acc)
            print(f"{step}/{params.epoch_size},reonstruction loss : {recon_loss:.2f}, disc_loss : {dis_loss:.2f}, disc_accuracy = {dis_acc.numpy()}, {round(time() - t, 2)}")
            

        history['reconstruction_loss'].append(np.mean(recon_loss_tab))
        history['discriminator_loss'].append(np.mean(dis_loss_tab))
        history['dis_accuracy'].append(np.mean(dis_accuracy_tab))

        # Validation
        recon_val_loss = []
        dis_val_loss = []
        dis_val_accuracy = []

        for step in range(train_indices, val_indices, eval_bs):
            t = time()
            stepTo = step + eval_bs if step +eval_bs < val_indices else val_indices 

            batch_x, batch_y = Data.load_batch_sequentially(step, stepTo)
            recon_loss, dis_loss, dis_acc = f.evaluate_on_val((batch_x, batch_y))

            recon_val_loss.append(recon_loss)
            dis_val_loss.append(dis_loss)
            dis_val_accuracy.append(dis_acc)

            print(f"{step- train_indices}/{val_indices - train_indices},reonstruction loss : {recon_loss:.2f}, disc_loss : {dis_loss:.2f}, disc_accuracy = {round(dis_acc.numpy(), 3)}, {round(time() - t, 2)}")


        history['reconstruction_val_loss'].append(np.mean(recon_val_loss))
        history['discriminator_val_loss'].append(np.mean(dis_val_loss))
        history['dis_val_accuracy'].append(np.mean(dis_val_accuracy))

       

    
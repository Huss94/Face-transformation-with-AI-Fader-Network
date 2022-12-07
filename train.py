#!/usr/bin/env python
import argparse
import pickle
from time import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Progbar

from loader import Loader
from model import AutoEncoder, Classifier, Fader
from utils import *

#L'utilisation du parser a été inspiré du code impélmenté par les développeur du fader network (bien que ce soit une chose commune)
parser = argparse.ArgumentParser(description='Train the fader Network')

parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images. It can be the directory of the image, or the npz file")
parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
parser.add_argument("--attr", type = str, default= "Male", help= "Considered attributes to train the network with")
parser.add_argument("--n_epoch", type = int, default = 1000, help = "Numbers of epochs")
parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")
parser.add_argument("--loading_mode", type = str, default = "preprocessed", help = "2 values : 'preprocessed' or 'direct'. from what the data are loaded npz file or direct data")
parser.add_argument("--load_in_ram", type= int, default = 0, help = "Si l'ordinateur n'a pas assez de ram pour charger toutes les données en meme temps, mettre False, le programme chargera seuleemnt les batchs de taille défini (32 par default) puis les déchargera après le calcul effectué") 
parser.add_argument("--save_path", type= str, default = "models", help = "Indique où enrisitrer le model") 
parser.add_argument("--classifier_path", type= str, default = 'models/classifier', help = 'path to the trained classifier if classifier is given (optional)')
parser.add_argument("--eval_bs", type= int, default = 32, help = 'Taille avec laquelle on subdivise la pase d\'évaluation')
parser.add_argument("--model_path", type= str, default = 'models/Fader_backup', help = "si on a déja entrainé un model, on peut continuer l'entrainment de model en spécifiant son chemin")
parser.add_argument("--h_flip", type = int, default =0, help = "Flip horizontalement les images (data aumgentation)")
parser.add_argument("--v_flip", type = int, default =0, help = "Flip verticalement les images (data aumgentation)")
parser.add_argument("--weighted", type = float, default = 0.3, help = "The probability (float between 0 and 1) in which we want to feed the Network  for the trained attributes. If 0 we use default dataset")

params = parser.parse_args()
assert params.weighted <= 1 and params.weighted >=0
if __name__ == "__main__":

    Data = Loader(params, weighted_attributes = params.weighted)


    train_indices = Data.train_indices

    #20000 pour évaluer c'est beaucoup trop, le temps de calcul serait trop long inutilmenet
    val_indices = Data.val_indices - 15000

    #eval_bs est le nombre de fichier a charger d'un coup dans la ram 
    eval_bs = params.eval_bs 

    # Une classe qu'on a créé afin de gérer les metrics
    metrics = Metrics('reconstruction_loss','discriminator_loss','dis_accuracy','reconstruction_val_loss','discriminator_val_loss' , 'dis_val_accuracy', 'classifier_loss', 'classifier_acc')

    # Création des models
    if params.model_path: 
        # Dans le cas où on continue l'entainement d'un model
        f = load_model(params.model_path, 'f')
        print("Model LOADED")
        metrics.load_dic_from_path(params.model_path)
        if len(metrics["reconstruction_val_loss"]) > 0:
            best_val_loss = metrics['reconstruction_val_loss'][-2]

            if f.params.classifier_path:
                best_val_acc = metrics['classifier_acc'][-2]
            else:
                best_val_acc = 0
        
        assert params.attr == f.params.attr
        Data = Loader(params, params.weighted)
    if not params.model_path:
        f = Fader(params)
        best_val_loss = np.inf
        best_val_acc = 0

    if params.classifier_path:
        # Si on donne un classifier, en effet le classifier n'est pas obligatoire
        C = load_model(params.classifier_path, model_type = 'c')
        C.training = False

    # Une erreur étrange sur le serveu gpu  de la fac qui demandait d'utiliser keras.optimizers.legacy.Adam
    try:
        f.compile( ae_opt= keras.optimizers.Adam(learning_rate=0.0002), dis_opt= keras.optimizers.Adam(learning_rate=0.0002), ae_loss = keras.losses.MeanSquaredError())
    except: 
        f.compile( ae_opt= keras.optimizers.legacy.Adam(learning_rate=0.0002), dis_opt= keras.optimizers.legacy.Adam(learning_rate=0.0002), ae_loss = keras.losses.MeanSquaredError())


    cur_epoch = len(metrics)
    # tf.config.run_functions_eagerly(True)



    #Boucle d'entrainement
    for epoch in range(cur_epoch, params.n_epoch):

        train_progbar = Progbar(params.epoch_size)
        eval_progbar = Progbar(val_indices - train_indices)

        print(f"Epoch {epoch} / {params.n_epoch}")
        print("Training")
        for step in range(0, params.epoch_size, params.batch_size):

            batch_x, batch_y = Data.load_random_batch(1, train_indices, params.batch_size)
            metrics_step = f.train_step((batch_x, batch_y))

            metrics.update(metrics_step)
            train_progbar.add(params.batch_size, values = dic_to_tab(metrics_step, exception = ["dis_accuracy"]))


        # Validation
        print("Evaluation")
        for step in range(train_indices, val_indices, eval_bs):
            stepTo = step + eval_bs if step +eval_bs < val_indices else val_indices 

            batch_x, batch_y = Data.load_batch_sequentially(step, stepTo)
            metrics_step = f.test_step((batch_x, batch_y))

            if params.classifier_path:
                metrics_step.update(C.eval_on_recons_attributes_batch((batch_x, tf.Variable(batch_y)), f))

            metrics.update(metrics_step)
            eval_progbar.add(eval_bs, values = dic_to_tab(metrics_step, exception= ["dis_val_accuracy", "classifier_acc", "classifier_loss"]))

        # On sauvegarde a chaque epoque le fader_network au cas ou la machine crash, on pourra reprendre l'entrainement
        # save_model_weights prend aussi en compte les poids des opimizers. et enregistre l'historique
        save_model_weights(f,   "Fader_backup", metrics.mean_dic,  params.save_path)

        # Sauvegarder le meilleur model a chaque epoch
        # On a 2 criètres pour la sauvegarde du model, celui qui reconstruit le mieux (plus petite reconstruciton loss)
        # Et celui dont le classifier entrainé en amont reconnait les attributs utilisé pour reconstruire l'image
        if metrics['reconstruction_val_loss'][-1] < best_val_loss:
            best_val_loss = metrics['reconstruction_val_loss'][-1]
            save_model_weights(f.ae, "Ae_best_loss",metrics.mean_dic, params.save_path)

        if params.classifier_path and metrics['classifier_acc'][-1] > best_val_acc:
            best_val_acc = metrics['classifier_acc'][-1]
            save_model_weights(f.ae, "Ae_best_acc",metrics.mean_dic, params.save_path)

        if epoch % 5 == 0: 
            try:
                save_model_weights(f.ae, params.attr[0] +'_'+ str(epoch), metrics.mean_dic, params.save_path)
            except:
                pass

        metrics.new_epoch()
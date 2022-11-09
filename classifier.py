from model import Classifier
import tensorflow as tf
from tensorflow import keras
from loader import Loader
import numpy as np
from utils import save_model, save_model_weights
from time import time
import argparse

parser = argparse.ArgumentParser(description='Trainong of the calssifier')
parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images")
parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
parser.add_argument("--attr", type = str, default= "*", help= "Considered attributes to train the network with")
parser.add_argument("--n_epoch", type = int, default = 5, help = "Numbers of epochs")
parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")
parser.add_argument("--loading_mode", type = str, default = "preprocessed", help = "2 values : 'preprocessed' or 'direct'. from what the data are loaded npz file or direct data")
parser.add_argument("--load_in_ram", type= bool, default = False, help = "Si l'ordinateur n'a pas assez de ram pour charger toutes les données en meme temps, mettre False, le programme chargera seuleemnt les batchs de taille défini (32 par default) puis les déchargera après le calcul effectué") 
parser.add_argument("--resize", type= bool, default = False, help = "Applique le resize a chaque fois qu'une donnée est chargée. Mettre a False si les images on été resized en amont") 
parser.add_argument("--save_path", type= str, default = "models", help = "Indique où enrisitrer le model") 


# Pour charger toutes les données en ram il faudrait environ 40 go de ram
# c'est pourquoi on preferera l'argument load_in_ram = false

params = parser.parse_args()
if __name__ == '__main__':

    # On reprend les paramètres utilisés par les auteurs de l'article
    train_indices = 162770
    val_indices = train_indices + 19867 

    # eval_bs correspond au batch a charger en mémooire pour l'évaluation, afin de pouvoir évaluer en plusieurs fois sur les petites configs
    eval_bs = 10 


    Data = Loader(params, train_indices, val_indices)

    C = Classifier(params)
    C.compile(optimizer= keras.optimizers.Adam(learning_rate=0.0002))

    history = {'train_loss' : [], 'train_acc': [], 'val_loss' : [], 'val_acc' : []}
    best_acc = 0
    # tf.config.run_functions_eagerly(True)

    print("Training:")
    for epoch in range(params.n_epoch):
        #training loop
        loss = []
        acc = []
        for step in range(0, params.epoch_size, params.batch_size):
            t = time()
            batch_x, batch_y  = Data.load_random_batch(1, train_indices, params.batch_size)
            l,a = C.train_step((batch_x, batch_y))

            loss.append(l)
            acc.append(a)
            print(f"epoch : {1 + epoch}/{params.n_epoch}, {step}/{params.epoch_size}, accuracy = {a.numpy():.2f}, loss = {l.numpy():.2f} calculé en : {time() - t:.2f}s")
                
        history['train_loss'].append(np.mean(loss))
        history['train_acc'].append(np.mean(acc))

        #Eval loop 
        loss = []
        acc = []
        print("Evaluation : ")
        for step in range(train_indices, val_indices, eval_bs):
            t = time()

            stepTo = step + eval_bs if step +eval_bs < val_indices else val_indices 

            batch_x, batch_y = Data.load_batch_sequentially(step,stepTo)
            l, a= C.eval_on_batch((batch_x, batch_y))
            loss.append(l)
            acc.append(a)

            print(f"{step- train_indices}/{val_indices - train_indices}, accuracy = {a.numpy():.2f}, {time() - t:.2f}")
        
        # Peut nous permettre de tracer un graph.
        history['val_loss'].append(np.mean(loss))
        history['val_acc'].append(np.mean(acc))

        if history['val_acc'][-1] > best_acc: 
            best_acc = history['val_acc'][-1]
            save_model_weights(C, name = 'classifier', folder_name=params.save_path)






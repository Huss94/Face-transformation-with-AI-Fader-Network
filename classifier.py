from model import Classifier
import tensorflow as tf
from tensorflow import keras
from loader import Loader
import numpy as np
from utils import save_model
from time import time
import argparse

parser = argparse.ArgumentParser(description='Trainong of the calssifier')
parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
parser.add_argument("--img_path", type = str, default = "data/img_align_celeba", help= "Path to images")
parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
parser.add_argument("--attr", type = str, default= "Smiling", help= "Considered attributes to train the network with")
parser.add_argument("--n_epoch", type = int, default = 5, help = "Numbers of epochs")
parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")

params = parser.parse_args()
if __name__ == '__main__':

    # On reprend les paramètres utilisés par les auteurs de l'article
    train_indices = 162770
    val_indices = train_indices + 19867


    Data = Loader(params, train_indices, val_indices)

    C = Classifier(params)
    C.compile(optimizer= keras.optimizers.Adam(learning_rate=0.0002))

    history = {'train_loss' : [], 'train_acc': [], 'val_loss' : [], 'val_acc' : []}
    best_acc = 0
    tf.config.run_functions_eagerly(True)

    for epoch in range(params.n_epoch):
        #training loop
        loss = []
        acc = []
        for step in range(0, params.epoch_size, params.batch_size):
            t = time()
            batch_x, batch_y  = Data.load_random_batch(0, train_indices, params.batch_size)
            l,a = C.train_step((batch_x, batch_y))
            loss.append(l)
            acc.append(a)
            print(step, a.numpy(), time() - t)
                
        history['train_loss'].append(np.mean(loss))
        history['train_acc'].append(np.mean(acc))

        #Eval loop 
        loss = []
        acc = []
        for step in range(train_indices, val_indices, params.batch_size):
            t = time()
            batch_x, batch_y = Data.load_batch_sequentially(step, step+params.batch_size)
            l, a= C.eval_on_batch((batch_x, batch_y))
            loss.append(l)
            acc.append(a)
            print(step, a, time() - t)
        
        # Peut nous permettre de tracer un graph.
        history['val_loss'].append(np.mean(loss))
        history['val_acc'].append(np.mean(acc))

        if history['val_acc'][-1] > best_acc: 
            best_acc = history['val_acc'][-1]
            save_model(C, 'classifier')






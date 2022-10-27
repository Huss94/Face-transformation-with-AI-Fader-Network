from re import L
from tensorflow import keras
from keras.layers import Input,Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, Reshape , ZeroPadding2D, LeakyReLU, BatchNormalization
from keras.models import Model,Sequential
import numpy as np 
import tensorflow as tf

n_layers = 7 

max_filters = 512
im_size = 256

def create_encoder(): 
    encoder = Sequential()

    for i in range(n_layers):
        # On respecte les indications données dans l'aritcles, a savoir, un padding de 1 
        # cur_im_size = im_size//(2**i)

        nb_filters = min(16*2**i, max_filters)
        if i == 0:
            encoder.add(ZeroPadding2D(padding=(1, 1), input_shape = [im_size, im_size, 3]))
        else: 
            encoder.add(BatchNormalization())
            encoder.add(ZeroPadding2D(padding=(1, 1)))

        encoder.add(Conv2D(nb_filters,(4,4), strides =(2,2), activation=LeakyReLU(alpha=0.2)))

    
    return encoder

def create_decoder(n_attr):
    decoder = Sequential()
    for i in range(n_layers)[::-1]:
        nb_filters = min(16*2**i, max_filters)
        print(i)

        if i == 6:
            decoder.add(Conv2DTranspose(nb_filters, kernel_size = (4, 4),padding= "same", strides = (2,2), activation='relu', input_shape = (2,2,512+n_attr)))
        elif i > 0: 
            decoder.add(Conv2DTranspose(nb_filters, kernel_size = (4, 4),padding = "same", strides = (2,2),activation='relu'))
            decoder.add(BatchNormalization())
        
        else: 
            decoder.add(Conv2DTranspose(3, kernel_size = (4, 4),padding = "same",  strides = (2,2), activation='relu', ))
            decoder.add(BatchNormalization())


    return decoder

train = np.load('data/test.npz')['arr_0']
e1 = train[0:1]

enc= create_encoder()
v= enc(e1)


dec = create_decoder(4)
dec.summary()
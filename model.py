from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, LeakyReLU, BatchNormalization, Dropout, Dense
from keras.models import Model,Sequential
import numpy as np 
import tensorflow as tf

n_layers = 7 

max_filters = 512
im_size = 256

def create_autoencoder(n_attr = 4): 
    encoder = Sequential(name = "encoder")
    encoder.add(keras.Input(shape=(im_size, im_size, 3)))

    decoder = Sequential(name = "decoder")
    decoder.add(keras.Input(shape = (2, 2, 512 + n_attr)))

    for i in range(n_layers):
        nb_filters_enc = min(16*2**i, max_filters)
        nb_filters_dec = min(16*2**(n_layers -(i+1)), max_filters)

        # Encoder
        encoder.add(Conv2D(nb_filters_enc, 4, 2, 'same', activation = LeakyReLU(alpha=0.2)))
        
        # Decoder
        if i == n_layers -1:
            decoder.add(Conv2DTranspose(3, 4, 2, 'same', activation='relu'))
        else:
            decoder.add(Conv2DTranspose(nb_filters_dec, 4, 2, 'same', activation='relu'))


        if i > 0:
            # BatchNorm avant ou apres la fonction d'activation ???? A tester
            # https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293/3

            encoder.add(BatchNormalization())
            decoder.add(BatchNormalization())

    return encoder, decoder

def create_discriminator(n_attr = 4):
    discriminator = Sequential(name = "discriminator")

    # The shape of the latent form
    discriminator.add(keras.Input(shape = (2,2,512)))

    discriminator.add(Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)))
    discriminator.add(BatchNormalization())
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512, activation=LeakyReLU(0.2)))
    discriminator.add(Dense(n_attr))

    return discriminator

class AutoEncoder(keras.Model):
    """
    La présence de cette classe est du au fait que le decoder a besoin de la represéntation latente z, et des attributs y pour reconstituer l'image avec l'attribut y 
    """
    def __init__(self, n_attr = 4):
        super(AutoEncoder, self).__init__()
        self.encoder, self.decoder = create_autoencoder(n_attr)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, y):
        # Le décodeur prend en entrée la concaténation de z et de y selon l'axe des colones
        y = np.expand_dims(y,(1, 2))
        y = np.repeat(y, 2, axis = 1)
        y = np.repeat(y, 2, axis = 2)
        zy = tf.concat((z,y), axis = -1)
        return self.decoder(zy)
        
    

    def call(self, x, y):
        z = self.encode(x)
        return z, self.decode(z, y)


class Fader(keras.Model):
    def __init__(self, autoencoder, discriminator, lambdae):
        super(Fader, self).__init__()
        self.ae = autoencoder
        self.discriminator = discriminator
        self.lambdae = lambdae
        self.n_iter = 0
        self.lambda_dis = 0
    

    def compile(self, autoenc_opt, dis_opt, ae_loss, dis_loss = tf.keras.losses.CategoricalCrossentropy()):
        self.dis_opt = dis_opt
        self.dis_loss = dis_loss
        self.ae_loss = ae_loss


    def train_step(self, data): 
        # On considère pour l'instant que x et y sont des numpy arrays
        x,y = data
        print(x.shape)

        #Training of the discriminator
        self.discriminator.trainable = True
        self.ae.trainable = False

        z = self.ae.encode(x)
        with tf.GradientTape as tape:
            y_preds = self.discriminator(z)
            dis_loss = self.dis_loss(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainabale = True

        with tf.GradientTape as tape:
            z, decoded = self.ae(x,decoded)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)


        self.n_iter+=1
        self.lambda_dis = 0.0001*min(self.n_iter/500000, 1)


if __name__  == "__main__":
    dis = create_discriminator(4)
    dis.summary()


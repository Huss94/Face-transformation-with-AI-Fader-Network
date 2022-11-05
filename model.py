from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, LeakyReLU, BatchNormalization, Dropout, Dense, Reshape
from keras.models import Model,Sequential
import numpy as np 
import tensorflow as tf

###
from torch.nn import functional as F
import torch


n_layers = 7 

max_filters = 512
im_size = 256


def attr_loss(y_true, y_preds, used_loss = tf.nn.softmax_cross_entropy_with_logits):
    bs = y_true.shape[0]
    n_attr = y_true.shape[-1]
    loss = 0
    # loss2 = 0

    for i in range(0,n_attr,2):
        yt = y_true[:, i:i+2]
        yp = y_preds[:, i : i+2]
        loss += tf.reduce_sum(used_loss(yt,yp))/bs
        
        # npr = torch.tensor(yp.numpy())
        # ntr = torch.tensor(yt.numpy())
        # loss2 += F.cross_entropy(npr, ntr)
        # print("loss : ", used_loss(yt,yp), "loss2", loss2)
        # loss3 = F.cross_entropy(npr, ntr[:, 1])
    return loss 

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
            # https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293/3

            encoder.add(BatchNormalization())
            decoder.add(BatchNormalization())

    return encoder, decoder

def create_discriminator(n_attr = 4):
    discriminator = Sequential(name = "discriminator")

    # The shape of the latent form
    discriminator.add(keras.Input(shape = (2,2,512)))

    discriminator.add(Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)))
    discriminator.add(BatchNormalization())
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512, activation=LeakyReLU(0.2)))
    discriminator.add(Dense(n_attr))
    discriminator.add(Reshape((n_attr,)))

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
        if type(y) != type(z):
            y = tf.constant(y)
        y = tf.expand_dims(y, axis = 1)
        y = tf.expand_dims(y, axis = 2)
        y = tf.repeat(y, 2, axis = 1)
        y = tf.repeat(y, 2, axis = 2)
        # y = np.expand_dims(y,(1, 2))
        # y = np.repeat(y, 2, axis = 1)
        # y = np.repeat(y, 2, axis = 2)
        zy = tf.concat((z,y), axis = -1)
        return self.decoder(zy)
        
    

    def call(self, x, y):
        z = self.encode(x)
        return z, self.decode(z, y)


class Fader(keras.Model):
    def __init__(self, autoencoder, discriminator = create_discriminator()):
        super(Fader, self).__init__()
        self.ae = autoencoder
        self.discriminator = discriminator
        self.n_iter = 0
        self.lambda_dis = 0
    

    def compile(self, ae_opt, dis_opt, ae_loss, dis_loss = attr_loss, run_eagerly = False):
        super(Fader,self).compile(run_eagerly = run_eagerly)
        self.run_eagerly =run_eagerly
        self.dis_opt = dis_opt
        self.ae_opt = ae_opt
        self.dis_loss = dis_loss
        self.ae_loss = ae_loss

    #Transformation en graph, accelere l'entrainemnet
    @tf.function
    def custom_train_step(self,data):
        """
        Cette méthode est la version de train_step customisée pour avoir le controole total sur le training (notamment les batch)
        """
        x,y = data
        #Training of the discriminator
        self.discriminator.trainable = True
        self.ae.trainable = False

        z = self.ae.encode(x)
        with tf.GradientTape() as tape:
            y_preds = self.discriminator(z)
            dis_loss = self.dis_loss(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as tape:
            z, decoded = self.ae(x,y)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)
            ae_loss = ae_loss + self.dis_loss(y, dis_preds)*self.lambda_dis
        grads = tape.gradient(ae_loss, self.ae.trainable_weights)
        self.ae_opt.apply_gradients(zip(grads, self.ae.trainable_weights))
            


        self.n_iter+=1
        self.lambda_dis = 0.0001*min(self.n_iter/500000, 1)
        return ae_loss, dis_loss

    def train_step(self, data): 
        # Cette fonciton est appelé a chaque step par model.fit()
        # On considère pour l'instant que x et y sont des numpy arrays
        x,y = data
        
        #Training of the discriminator
        self.discriminator.trainable = True
        self.ae.trainable = False

        z = self.ae.encode(x)
        with tf.GradientTape() as tape:
            y_preds = self.discriminator(z)
            dis_loss = self.dis_loss(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as tape:
            z, decoded = self.ae(x,y)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)
            ae_loss = ae_loss + self.dis_loss(y, dis_preds)*self.lambda_dis
        grads = tape.gradient(ae_loss, self.ae.trainable_weights)
        self.ae_opt.apply_gradients(zip(grads, self.ae.trainable_weights))
            


        self.n_iter+=1
        self.lambda_dis = 0.0001*min(self.n_iter/500000, 1)
        return {"reconstruction_loss": ae_loss, "dis_loss": dis_loss}

        
    
if __name__  == "__main__":
    dis = create_discriminator(4)
    dis.summary()


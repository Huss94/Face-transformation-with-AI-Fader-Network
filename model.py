from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, LeakyReLU, BatchNormalization, Dropout, Dense, Reshape
from keras.models import Model,Sequential
import numpy as np 
import tensorflow as tf
n_layers = 7 

max_filters = 512
im_size = 256

def compute_accuracy(yt,yp):
    count = 0
    for i in range(len(yp)):
        if tf.argmax(yp[i]) == tf.argmax(yt[i]):
            count+=1
    return count/len(yp) 


def attr_loss_accuracy(y_true, y_preds, used_loss = tf.nn.softmax_cross_entropy_with_logits):
    bs = y_true.shape[0]
    n_attr = y_true.shape[-1]
    loss = 0
    accuracy = []

    for i in range(0,n_attr,2):
        yt = y_true[:, i:i+2]
        yp = y_preds[:, i : i+2]
        loss += tf.reduce_sum(used_loss(yt,yp))/bs
        accuracy.append(compute_accuracy(yt, yp))


    return loss, tf.reduce_mean(accuracy)

def create_autoencoder(n_attr = 4): 
    encoder = Sequential()
    encoder.add(keras.Input(shape=(im_size, im_size, 3)))

    decoder = Sequential()
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

def create_discriminator(n_attr):
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

class Classifier(keras.Model):
    def __init__(self, params):
        self.params = params 
        super(Classifier, self).__init__()
        self.model, _  = create_autoencoder(0)
        self.model.add(Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, activation = LeakyReLU(0.2)))
        self.model.add(Dense(2*params.n_attr))
        self.model.add(Reshape((2*params.n_attr,)))
        self.build((None, 256,256,3))

    def compile(self, optimizer, loss = attr_loss_accuracy):
        super(Classifier, self).compile()
        self.opt = optimizer
        self.loss = loss

    @tf.function
    def eval_on_batch(self, data):
        x,y = data

        self.model.trainable = False
        y_preds = self.model(x)
        loss , acc= self.loss(y, y_preds)
        return loss, acc


    @tf.function
    def train_step(self, data):
        x,y = data
        self.model.trainable = True
        with tf.GradientTape() as tape:
            y_preds = self(x)
            loss , acc= self.loss(y, y_preds)

        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return loss, acc


    def call(self, x):
        x = self.model(x)
        return x

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

        # Pour certaine raison, le graph (eagerly mode = False) n'accepte pas le numpy array dans cette methode, on transform alors y en tenseur
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
    def __init__(self, params):
        super(Fader, self).__init__()
        self.ae = create_autoencoder(params.n_attr)
        self.discriminator = create_discriminator(params.n_attr)
        self.n_iter = 0
        self.lambda_dis = 0
    

    def compile(self, ae_opt, dis_opt, ae_loss, dis_loss = attr_loss_accuracy, run_eagerly = False):
        super(Fader,self).compile(run_eagerly = run_eagerly)
        self.run_eagerly =run_eagerly
        self.dis_opt = dis_opt
        self.ae_opt = ae_opt
        self.dis_loss = dis_loss
        self.ae_loss = ae_loss

    @tf.function
    def evaluate_on_val(self,data):
        x,y = data
        self.discriminator.trainable = False
        self.ae.trainable = False
        z, decoded = self.ae(x,y)
        y_preds = self.discriminator(z)

        #Discriminator
        dis_loss, dis_accuracy = self.dis_loss(y, y_preds)

        # Autoencoder
        ae_loss = self.ae_loss(x, decoded)
        ae_loss = ae_loss + dis_loss*self.lambda_dis

        return ae_loss, dis_loss, dis_accuracy

    #Transformation en graph, accelere l'entrainemnet
    @tf.function
    # Cette fonction peut s'apperler en utilisant model.fit mais on a préférer créer notre boucle d'entrainement personalisé danas main (notamment pour avoir le controle sur le chargement des données et donc la mémoire RAM)
    def  train_step(self,data):
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
            dis_loss ,dis_accuracy = self.dis_loss(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as tape:
            z, decoded = self.ae(x,y)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)
            ae_loss = ae_loss + self.dis_loss(y, dis_preds)[0]*self.lambda_dis
        grads = tape.gradient(ae_loss, self.ae.trainable_weights)
        self.ae_opt.apply_gradients(zip(grads, self.ae.trainable_weights))
            


        self.n_iter+=1
        self.lambda_dis = 0.0001*min(self.n_iter/500000, 1)
        return ae_loss, dis_loss, dis_accuracy

   
if __name__  == "__main__":
    dis = create_discriminator(4)
    dis.summary()


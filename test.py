import tensorflow as tf
import numpy as np 
from tensorflow import keras
from keras import layers

c = tf.saved_model.load("models/classifier")
print(c.compile)





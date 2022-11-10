import numpy as np 
import tensorflow as tf
from utils import * 
from loader import Loader
from model import Classifier
from time import time

tf.config.run_functions_eagerly(True)
C = load_model("models/trained_classifier", model_type  = 'c') 
C.compile(optimizer=None)
params = C.params
params.attr_path = 'data/attributes.npz'
params.img_path  = 'data/img_align_celeba_resized'

Data = Loader(params)
t = time()
x,y = Data.load_random_batch(Data.val_indices, Data.test_indices, 100)
print(time() - t)
loss, acc = C.eval_on_batch((x,y))

print(loss, acc)


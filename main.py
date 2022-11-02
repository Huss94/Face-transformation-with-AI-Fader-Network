import numpy as np 
import tensorflow as tf
from model import AutoEncoder, Disciminator




if __name__ == "__main__":
    # dis = Disciminator()
    # autoEnc = AutoEncoder()

    # Chargement des données 

    all_data = np.load("data/test.npz")['arr_0']/255
    all_data = all_data[:600]
    attributes = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item()

    train_indices = 500
    val_indices = 600

    x_train = all_data[:train_indices]
    x_val = all_data[train_indices:val_indices]

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(32)

    for d in dataset:
        ...


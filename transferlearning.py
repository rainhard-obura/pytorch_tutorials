import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hu as hub

(x_train, y_train), (x_test, y_test) = mnist.load_data()

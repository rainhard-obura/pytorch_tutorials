import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshpe(-1, 28 * 28).astype('float32') / 255.0

model = keras.models.load_model('pretrained')

#Freeze all model layer weights
model.trainable = False

# can also set trainable for specific layers
for layer in model.layers:
    #assert should be true because of the one-liner above
    assert layer.trainable == False
    layer.trainable = False
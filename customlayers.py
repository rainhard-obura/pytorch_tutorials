import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation ='relu')
        self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        return self.dense2(x)

model = MyModel()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
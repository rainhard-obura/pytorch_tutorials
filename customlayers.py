import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

class Dense(layers.Layer):
    def __init__(self, units, input_dim):
        super(Dense, self).__init__()
        self.w = self.add_weight(
            name = 'w',
            shape = (input_dim, units)
            initializer ='random_normal'
            trainable = True
        )
        self.b = self.add_weight(
            name='b', shape =(units,), initializer='zeros', trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),
            initializer = 'random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            name ='b', shape = (self.units, ), initializer='random_normal',
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return math.maximum(x, 0)

class MyModel(Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyReLU()

    def call(self, x):
        x = self.relu(self.dense1(x))
        return self.dense2(x)


model = MyModel()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
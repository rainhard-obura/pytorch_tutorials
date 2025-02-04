from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
)

from tensorflow.keras import Model
import tensorflow as tf
import typing

@tf.function

def AlexNet(input_shape: typing.Tuple[int], classes:int=1000) -> Model:

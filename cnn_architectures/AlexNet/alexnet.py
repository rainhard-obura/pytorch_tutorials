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
    '''
    Implementation of the AlexNet architecture

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    Returns -- a Model() instance in keras

    Note:
    when you read the paper, you will notice that the channels(filters) in the diagram is only
    half of what i have written below.That is because in the diagram, they only showed model for
    one GPU (I guess for simplicity). However, during the ILSVRC, they run the network across 2 NVIDIA GTA 580 3GB GPUs.

    Also, in paper, they used Local Response Normalization. This can also be done in Keras with Lambda layer.
    You can also use BatchNormalization layer instead.
    '''

    # convert input shape to tensor
    X_input = Input(input_shape)
    #NOTE:Layer 1-5 is conv layers
    # layer 1
    X = Conv2D(
        filters = 96,
        kernel_size = (11, 11),
        strides = (4, 4),
        activation = "relu",
        padding = "same",
    )(X_input)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    X = Lambda(tf.nn.local_response_normalization)(X)

    # layer 2
    X = Conv2D(
        filters = 256,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    X = Lambda(tf.nn.local_response_normalization)(X)

    # layer 3
    X = Conv2D(
        filters = 384,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)

    # layer 4
    X = Conv2D(
        filters = 384,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)

    # layer 5
    X = Conv2D(
        filters = 256,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)
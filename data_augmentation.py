import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train','test'], shuffle_files=True,
                                         as_supervised=True, with_info=True)

def normalize_img(image, label):
    ''' Normalizes images '''
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def augment(image, label):
    new_height = new_width = 32
    image = tf.image.resize(image, (new_height, new_width))

    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower = 0.1, upper = 0.2)

    # a left upside down flipped is still a dog
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.rando_flip_up_down(image)

    return image, label

# set up for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls = AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Set up for test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

from keras import layers
data_augmentation = keras.Sequential(
    [
        layers.Resizing(height=32, width=32,),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(factor=0.1,),
    ]
)

model = keras.Sequential([
    keras.Input((32,32,3)),
    data_augmentation,
    layers.Conv2D(4,3,padding='same', activation='relu'),
    layers.Conv2D(8,3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10),
])

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=3e-4)  # Define the optimizer properly

model.compile(
    optimizer = Adam(learning_rate=3e-4),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.fit(ds_train, epochs = 5, verbose = True)
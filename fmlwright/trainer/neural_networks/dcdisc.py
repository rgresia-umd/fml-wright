import tensorflow as tf
from tensorflow.keras import (layers, Sequential,)
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Dense,
    Flatten,
)

import logging

log = logging.getLogger(__name__)

def create_dcdisc(input_shape, dropout):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1))

    return model


import tensorflow as tf
from tensorflow.keras import (layers, Sequential,)
from tensorflow.keras.layers import (
    Conv2DTranspose,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Dense,
    Flatten,
    Reshape,
)

import logging

log = logging.getLogger(__name__)

def create_dcgen(input_shape, noise_dim, dropout):
    model = Sequential()
    model.add(Dense((input_shape[0]/4)*(input_shape[1]/4)*(input_shape[2]*256), use_bias=False, input_shape=(noise_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((int(input_shape[0]/4), int(input_shape[1]/4), int(input_shape[2]*256))))
    assert model.output_shape == (None, int(input_shape[0]/4), int(input_shape[1]/4), int(input_shape[2]*256))  # Note: None is the batch size

    model.add(Conv2DTranspose(input_shape[2]*128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(input_shape[0]/4), int(input_shape[1]/4), int(input_shape[2]*128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(input_shape[2]*64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(input_shape[0]/2), int(input_shape[1]/2), int(input_shape[2]*64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(input_shape[2], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))

    return model

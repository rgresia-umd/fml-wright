
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2DTranspose,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Sequential,
    Dense,
    Flatten,
    Reshape,
)

import logging

log = logging.getLogger(__name__)

def create_dcgen(input_shape, output_shape, dropout):
    model = Sequential()
    model.add(Dense((output_shape[0]/4)*(output_shape[1]/4)*(output_shape[2]*256), use_bias=False, input_shape=input_shape)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape(((output_shape[0]/4), (output_shape[1]/4), (output_shape[2]*256))))
    assert model.output_shape == (None, (output_shape[0]/4), (output_shape[1]/4), (output_shape[2]*256))  # Note: None is the batch size

    model.add(Conv2DTranspose(output_shape[2]*128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, (output_shape[0]/4), (output_shape[1]/4), (output_shape[2]*128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(output_shape[2]*64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, (output_shape[0]/2), (output_shape[1]/2), (output_shape[2]*64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(output_shape[2], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, (output_shape[0]), (output_shape[1]), (output_shape[2]))

    return model

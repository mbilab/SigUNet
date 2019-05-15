#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Input, Concatenate
from tensorflow.keras.models import Model

from .layers import Conv1DTranspose

class SigUNet(Model):

    def __init__(self, m, n, kernel_size):
        super(SigUNet, self).__init__()

        self._conv_layers = [
            Conv1D(filters=m, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m+n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m+n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m+2*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m+2*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m+3*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m+3*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m+2*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m+2*n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m+n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m+n, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),

            Conv1D(filters=m, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=m, kernel_size=kernel_size, padding='same', activation=tf.nn.relu),
            Conv1D(filters=3, kernel_size=1, padding='same', activation=tf.nn.softmax),
        ]

        self._deconv_layers = [
            Conv1DTranspose(filters=m+2*n, kernel_size=kernel_size, stride=2),
            Conv1DTranspose(filters=m+n, kernel_size=kernel_size, stride=2),
            Conv1DTranspose(filters=m, kernel_size=kernel_size, stride=2),
        ]

    def call(self, x):

        skip_conn = []
        output = tf.one_hot(x, depth=20)

        output = self._conv_layers[0](output)
        output = self._conv_layers[1](output)
        skip_conn.append(output)
        output = AveragePooling1D(pool_size=2)(output)

        output = self._conv_layers[2](output)
        output = self._conv_layers[3](output)
        skip_conn.append(output)
        output = AveragePooling1D(pool_size=2)(output)

        output = self._conv_layers[4](output)
        output = self._conv_layers[5](output)
        skip_conn.append(output)
        output = AveragePooling1D(pool_size=2)(output)

        output = self._conv_layers[6](output)
        output = self._conv_layers[7](output)
        output = self._deconv_layers[0](output)

        output = Concatenate()([output, skip_conn[-1]])

        output = self._conv_layers[8](output)
        output = self._conv_layers[9](output)
        output = self._deconv_layers[1](output)

        output = Concatenate()([output, skip_conn[-2]])

        output = self._conv_layers[10](output)
        output = self._conv_layers[11](output)
        output = self._deconv_layers[2](output)

        output = Concatenate()([output, skip_conn[-3]])

        output = self._conv_layers[12](output)
        output = self._conv_layers[13](output)
        output = self._conv_layers[14](output)

        return output

def get_model(input_layer=None):

    def wrapper(n, m, kernel_size):

        x = Input(shape=(96,), dtype=tf.int32) if input_layer is None else input_layer
        logits = SigUNet(n, m, kernel_size)(x)

        model = Model(inputs=x, outputs=logits)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        return model

    return wrapper

def load_model(config, weights, input_layer=None):
    model = get_model(input_layer)(**config)
    model.load_weights(weights)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model

if __name__ == '__main__':

    import numpy as np

    data = np.load('../data/features/train.npy')
    x = data['features']
    y = data['residue_label']
    model = get_model(n=12, m=6, kernel_size=11)

    model.fit(x, y, batch_size=96, epochs=10, validation_split=0.2)

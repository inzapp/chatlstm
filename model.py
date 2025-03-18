"""
Authors : inzapp

Github url : https://github.com/inzapp/chatlstm

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, cfg, max_sequence_length, vocab_size):
        self.cfg = cfg
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.sequence_downscaling_target = 64

    def build(self):
        input_layer = tf.keras.layers.Input(shape=(self.max_sequence_length,))
        x = input_layer
        x = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.cfg.embedding_dim)(x)
        conv_filters = self.cfg.embedding_dim * 2
        sequence_length = self.max_sequence_length
        for _ in range(5):
            if sequence_length <= self.sequence_downscaling_target:
                strides = 1
            else:
                strides = 2
                sequence_length /= 2
            x = self.conv1d(x, conv_filters, 5, strides, activation='leaky')
            conv_filters = min(conv_filters * 2, 4096)
        x = self.conv1d(x, self.cfg.recurrent_units, 1, 1, activation='leaky')
        if self.cfg.use_gru:
            x = self.gru(x, units=self.cfg.recurrent_units)
        else:
            x = self.lstm(x, units=self.cfg.recurrent_units)
        output_layer = self.output_layer(x)
        model = tf.keras.models.Model(input_layer, output_layer)
        return model

    def conv1d(self, x, filters, kernel_size, strides, activation='leaky'):
        x = tf.keras.layers.Conv1D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def lstm(self, x, units, initial_state=None, return_state=False):
        if return_state:
            x, state_h, state_c = tf.keras.layers.LSTM(units=units, return_state=return_state)(x, initial_state=initial_state)
            return x, [state_h, state_c]
        else:
            x = tf.keras.layers.LSTM(units=units, return_state=return_state)(x, initial_state=initial_state)
            return x

    def gru(self, x, units, initial_state=None, return_state=False):
        if return_state:
            x, state_h = tf.keras.layers.GRU(units=units, return_state=return_state)(x, initial_state=initial_state)
            return x, [state_h]
        else:
            x = tf.keras.layers.GRU(units=units, return_state=return_state)(x, initial_state=initial_state)
            return x

    def calculate_units(self, c=3, min_units=64, max_units=1024):
        return min(max(int(c * math.sqrt(self.vocab_size)), min_units), max_units)

    def output_layer(self, x):
        # x = tf.keras.layers.Dense(units=self.calculate_units(), kernel_initializer=self.kernel_initializer())(x)
        # x = tf.keras.layers.Dense(units=1024, kernel_initializer=self.kernel_initializer())(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = self.activation(x, 'leaky')
        if self.cfg.dropout < 0.0 <= 1.0:
            x = tf.keras.layers.Dropout(self.cfg.dropout)(x)
        return tf.keras.layers.Dense(units=self.vocab_size, kernel_initializer=self.kernel_initializer(), activation='softmax')(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def kernel_regularizer(self, l2=0.0005):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation, name=None):
        if activation == 'linear':
            return x
        elif activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x


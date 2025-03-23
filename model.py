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
    def __init__(self, cfg, model_input_sequence_length, vocab_size):
        self.cfg = cfg
        self.model_input_sequence_length = model_input_sequence_length
        self.vocab_size = vocab_size
        self.sequence_downscaling_target = 64

    def min_division_step(self, n, target, max_step=10):
        for i in range(max_step):
            if n / (2 ** i) <= target:
                return i
        return max_step

    def build(self):
        input_layer = tf.keras.layers.Input(shape=((self.model_input_sequence_length,)))
        x = input_layer
        x = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.cfg.embedding_dim)(x)
        conv_filters = self.cfg.embedding_dim * 2
        sequence_length = self.model_input_sequence_length
        for _ in range(max(1, self.min_division_step(self.model_input_sequence_length, self.sequence_downscaling_target))):
            if sequence_length <= self.sequence_downscaling_target:
                strides = 1
            else:
                strides = 2
                sequence_length /= 2
            x = self.conv1d(x, conv_filters, 5, strides, activation='leaky')
            conv_filters = min(conv_filters * 2, self.cfg.max_conv_filters)
        x = self.conv1d(x, self.cfg.recurrent_units, 1, 1, activation='leaky')
        x = self.multi_head_attention(x, num_heads=4, key_dim=self.cfg.recurrent_units)
        if self.cfg.use_gru:
            x = self.gru(x, units=self.cfg.recurrent_units)
            x = self.gru(x, units=self.cfg.recurrent_units)
            x = self.gru(x, units=self.cfg.last_recurrent_units)
        else:
            x = self.lstm(x, units=self.cfg.recurrent_units)
            x = self.lstm(x, units=self.cfg.recurrent_units)
            x = self.lstm(x, units=self.cfg.last_recurrent_units)
        x = tf.keras.layers.Flatten()(x)
        output_layer = self.output_layer(x)
        model = tf.keras.models.Model(input_layer, output_layer)
        return model

    def ln(self, x):
        return tf.keras.layers.LayerNormalization()(x)

    def conv1d(self, x, filters, kernel_size, strides, activation='leaky'):
        x = tf.keras.layers.Conv1D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer())(x)
        return self.activation(x, activation)

    def lstm(self, x, units):
        x = tf.keras.layers.LSTM(
            units=units,
            kernel_regularizer=self.kernel_regularizer(),
            recurrent_regularizer=self.kernel_regularizer(),
            return_sequences=True)(x)
        return x

    def gru(self, x, units):
        x = tf.keras.layers.GRU(
            units=units,
            kernel_regularizer=self.kernel_regularizer(),
            recurrent_regularizer=self.kernel_regularizer(),
            return_sequences=True)(x)
        return x

    def multi_head_attention(self, x, num_heads, key_dim, dropout=0.0):
        x_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
        x = tf.keras.layers.Add()([x_attention, x])
        x = self.ln(x)
        return x

    def output_layer(self, x):
        if self.cfg.dropout < 0.0 <= 1.0:
            x = tf.keras.layers.Dropout(self.cfg.dropout)(x)
        return tf.keras.layers.Dense(units=self.vocab_size, kernel_initializer=self.kernel_initializer(), activation='softmax')(x)

    def kernel_initializer(self):
        return tf.keras.initializers.he_normal() if self.cfg.l2 > 0.0 else None

    def kernel_regularizer(self, l2=0.0005):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation, name=None):
        if activation == 'linear':
            return x
        elif activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x


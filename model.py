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
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, max_sequence_length, vocab_size, embedding_dim, recurrent_units, use_gru):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.recurrent_units = recurrent_units
        self.use_gru = use_gru
        self.num_downsacling = int(np.ceil(self.max_sequence_length / 64)) - 1

    def build(self):
        encoder_input, encoder_states = self.build_encoder()
        decoder_input, decoder_output = self.build_decoder(encoder_states)
        model = tf.keras.models.Model([encoder_input, decoder_input], decoder_output)
        return model

    def build_encoder(self):
        input_layer = tf.keras.layers.Input(shape=(self.max_sequence_length,))
        x = input_layer
        x = self.embedding(x, input_dim=self.vocab_size, output_dim=self.embedding_dim)

        if self.num_downsacling == 0:
            x = self.conv1d(x, self.embedding_dim, 5, 1, activation='relu')
        else:
            for _ in range(self.num_downsacling):
                x = self.conv1d(x, self.embedding_dim, 5, 2, activation='relu')

        if self.use_gru:
            x, states = self.gru(x, units=self.recurrent_units, return_state=True)
        else:
            x, states = self.lstm(x, units=self.recurrent_units, return_state=True)
        return input_layer, states

    def build_decoder(self, encoder_states):
        input_layer = tf.keras.layers.Input(shape=(self.max_sequence_length,))
        x = input_layer
        x = self.embedding(x, input_dim=self.vocab_size, output_dim=self.embedding_dim)

        if self.num_downsacling == 0:
            x = self.conv1d(x, self.embedding_dim, 5, 1, activation='relu')
        else:
            for _ in range(self.num_downsacling):
                x = self.conv1d(x, self.embedding_dim, 5, 2, activation='relu')

        if self.use_gru:
            x = self.gru(x, units=self.recurrent_units, initial_state=encoder_states)
        else:
            x = self.lstm(x, units=self.recurrent_units, initial_state=encoder_states)
        output_layer = self.output_layer(x)
        return input_layer, output_layer

    def embedding(self, x, input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)(x)

    def conv1d(self, x, filters, kernel_size, strides, activation='relu'):
        x = tf.keras.layers.Conv1D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer())(x)
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

    def output_layer(self, x):
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


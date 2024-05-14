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
import os
import csv
import konlpy
import numpy as np
import tensorflow as tf

from tqdm import tqdm


class DataGenerator:
    def __init__(self,
                 data_path,
                 batch_size,
                 pretrained_maxlen=0,
                 pretrained_vocab_size=0,
                 evaluate=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.evaluate = evaluate
        self.x_datas = []
        self.y_datas = []
        self.pretrained_maxlen = pretrained_maxlen
        self.pretrained_vocab_size = pretrained_vocab_size
        self.__maxlen = 0
        self.__vocab_size = 0
        self.morph_analyzer = None
        self.tokenizer = None
        self.start_token = None
        self.end_token = None
        self.__start_sequence = None
        self.prepared = False

    def is_valid_path(self, path):
        return os.path.exists(path) and path.lower().endswith('.csv')

    def get_vocab_size(self):
        assert self.prepared
        return self.__vocab_size

    def get_maxlen(self):
        assert self.prepared
        return self.__maxlen

    def get_start_sequence(self):
        assert self.prepared
        return self.__start_sequence

    def pad_sequences(self, sequences):
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.__maxlen, padding='post')

    def prepare(self):
        if self.prepared:
            return

        if not self.is_valid_path(self.data_path):
            print(f'data path is invalid => [{self.data_path}]')
            exit(0)

        csv_lines = 0
        with open(self.data_path, 'rt') as f:
            csv_lines = len(f.readlines())

        morphed_nls = []
        self.morph_analyzer = konlpy.tag.Okt()
        with open(self.data_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, total=csv_lines):
                assert len(row) >= 2, 'csv must have at least two columns for input and output pair.'
                q = row[0]
                a = row[1]
                morphed_nls.append(self.morph_analyzer.morphs(q))
                morphed_nls.append(self.morph_analyzer.morphs(a))

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(morphed_nls)
        sequences = self.tokenizer.texts_to_sequences(morphed_nls)

        data_maxlen = 0
        for s in sequences:
            data_maxlen = max(data_maxlen, len(s))
            if self.pretrained_maxlen > 0:
                assert data_maxlen <= self.pretrained_maxlen, f'data_maxlen({data_maxlen}) must be lower equal than given pretrained_maxlen({self.pretrained_maxlen})'
        self.__maxlen = max(data_maxlen, self.pretrained_maxlen)

        data_vocab_size = len(self.tokenizer.index_word) + 3  # 3 for [zero_pad, start_token, end_token]
        if self.pretrained_vocab_size > 0:
            assert data_vocab_size <= self.pretrained_vocab_size, 'data_vocab_size({data_vocab_size}) must be lower equal than given pretrained_vocab_size({self.pretrained_vocab_size})'
            self.__vocab_size = min(data_vocab_size, self.pretrained_vocab_size)
        else:
            self.__vocab_size = data_vocab_size

        self.start_token = max(data_vocab_size, self.pretrained_vocab_size) - 2
        self.end_token = max(data_vocab_size, self.pretrained_vocab_size) - 1

        self.__start_sequence = np.zeros(shape=(1, self.__maxlen), dtype=np.int32)
        self.__start_sequence[0][0] = self.start_token

        self.x_sequences = []
        self.y_sequences = []
        for i in range(len(sequences) // 2):
            x_sequence = sequences[i*2]
            y_sequence = sequences[i*2+1]
            assert len(x_sequence) > 0 and len(y_sequence) > 0, 'sequence length cannot be zero'
            self.x_sequences.append(x_sequence)
            self.y_sequences.append(y_sequence)
        assert len(self.x_sequences) == len(self.y_sequences)
        self.prepared = True

    def load(self):
        assert self.prepared
        encoder_batch_x, decoder_batch_x, batch_y = [], [], []
        indices = np.random.choice(len(self.x_sequences), self.batch_size, replace=False)
        for i in indices:
            x_sequence = self.x_sequences[i]
            y_sequence = self.y_sequences[i]
            encoder_batch_x.append(self.pad_sequences([x_sequence])[0])
            random_index = np.random.randint(len(y_sequence) + 1)
            if random_index == 0:
                decoder_x = [self.start_token]
                y = y_sequence[0]
            elif random_index == len(y_sequence):
                decoder_x = [self.start_token] + y_sequence
                y = self.end_token
            else:
                decoder_x = [self.start_token] + y_sequence[:random_index]
                y = y_sequence[random_index]
            decoder_batch_x.append(self.pad_sequences([decoder_x])[0])
            batch_y.append(y)
        encoder_batch_x = np.asarray(encoder_batch_x).astype(np.int32)
        decoder_batch_x = np.asarray(decoder_batch_x).astype(np.int32)
        batch_y = np.asarray(batch_y).astype(np.int32)
        return [encoder_batch_x, decoder_batch_x], batch_y

    def evaluate_generator(self):
        assert self.prepared
        for _ in range(len(self.x_sequences) // self.batch_size):
            yield self.load()

    def preprocess(self, nl):
        assert self.prepared
        morphed_nl = self.morph_analyzer.morphs(nl)
        sequence = self.tokenizer.texts_to_sequences([morphed_nl])[0]
        sequence_padded = self.pad_sequences([sequence])[0]
        x = np.asarray(sequence_padded).astype(np.int32)
        x = x.reshape((1,) + x.shape)
        return x

    def postprocess(self, y):
        assert self.prepared
        index = np.argmax(y)
        end = index in [0, self.end_token]
        if end:
            return 0, '<EOS>', end
        else:
            return index, self.tokenizer.index_word[index], end


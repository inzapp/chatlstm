"""
Authors : inzapp

Github url : https://github.com/inzapp/chat-lstm

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
                 maxlen=0,
                 vocab_size=0,
                 evaluate=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.evaluate = evaluate
        self.x_datas = []
        self.y_datas = []
        self.__maxlen = maxlen
        self.__vocab_size = vocab_size
        self.morph_analyzer = None
        self.tokenizer = None
        self.prepared = False

    def is_valid_path(self, path):
        return os.path.exists(path) and path.lower().endswith('.csv')

    def get_vocab_size(self):
        assert self.prepared
        return self.__vocab_size

    def get_maxlen(self):
        assert self.prepared
        return self.__maxlen

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
            tqdm_reader = tqdm(reader, total=csv_lines, unit='rows', mininterval=1)
            for row in tqdm_reader:
                assert len(row) >= 2, 'csv must have at least two columns for input and output pair.'
                q = row[0]
                a = row[1]
                morphed_nls.append(self.morph_analyzer.morphs(q))
                morphed_nls.append(self.morph_analyzer.morphs(a))

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(morphed_nls)
        sequences = self.tokenizer.texts_to_sequences(morphed_nls)

        maxlen = 0
        for s in sequences:
            maxlen = max(maxlen, len(s))
            if self.__maxlen > 0:
                assert maxlen <= self.__maxlen, f'data_maxlen({maxlen}) must be lower equal than given maxlen({self.__maxlen})'
        if self.__maxlen == 0:
            self.__maxlen = maxlen

        vocab_size = len(self.tokenizer.word_index) + 1
        if self.__vocab_size == 0:
            self.__vocab_size = vocab_size
        else:
            assert vocab_size <= self.__vocab_size, 'data_vocab_size({vocab_size}) must be lower equal than given vocab_size({vocab_size})'

        self.x_datas = []
        self.y_datas = []
        sequences_padded = self.pad_sequences(sequences)
        for i in range(len(sequences_padded) // 2):
            self.x_datas.append(sequences_padded[i*2])
            self.y_datas.append(sequences_padded[i*2+1])
        self.x_datas = np.asarray(self.x_datas).astype(np.float32)
        self.y_datas = np.asarray(self.y_datas).astype(np.float32)
        assert len(self.x_datas) == len(self.y_datas)
        self.prepared = True

    def load(self):
        indices = np.random.choice(len(self.x_datas), self.batch_size, replace=False)
        batch_x = np.asarray([self.x_datas[i] for i in indices]).astype(np.float32)
        batch_y = np.asarray([self.y_datas[i] for i in indices]).astype(np.float32)
        return batch_x, batch_y

    def preprocess(self, nl):
        morphed_nl = self.morph_analyzer.morphs(nl)
        sequence = self.tokenizer.texts_to_sequences(morphed_nl)
        sequence_padded = self.pad_sequences(sequence)
        x = np.asarray(sequence_padded).astype(np.float32)
        return x

    def postprocess(self, y):
        output_sequences = []
        for i in range(len(y)):
            index = np.argmax(y[i])
            output_sequences.append(index)
            if index == 0:
                break
        nl = self.tokenizer.sequences_to_texts([output_sequences])[0]
        return nl


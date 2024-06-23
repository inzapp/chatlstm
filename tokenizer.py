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


class Tokenizer:
    def __init__(self):
        self.tokens = []
        self.index_word = {}
        self.word_index = {}
        self.vocab_size = 0
        self.max_sequence_length = 0
        self.pad_token = '([<PAD>])'
        self.unk_token = '([<UNK>])'
        self.bos_token = '([<BOS>])'
        self.eos_token = '([<EOS>])'
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.init()

    def remove_special_tokens(self, nl):
        while True:
            cnt = 0
            for special_token in self.special_tokens:
                if nl.find(special_token) > -1:
                    nl = nl.replace(special_token, '')
                else:
                    cnt += 1
            if cnt == len(self.special_tokens):
                break
        return nl

    def add_token(self, token):
        if not (token in self.word_index):
            self.tokens.append(token)
            self.index_word[self.vocab_size] = token
            self.word_index[token] = self.vocab_size
            self.vocab_size += 1

    def init(self):
        self.tokens = []
        self.index_word = {}
        self.word_index = {}
        self.vocab_size = 0
        self.max_sequence_length = 0
        for special_token in self.special_tokens:
            self.add_token(special_token)

    def fit_on_texts(self, words_list):
        token_set = set()
        for i in range(len(words_list)):
            sequence_length = len(words_list[i]) + 2  # for bos, eos token
            self.max_sequence_length = max(sequence_length, self.max_sequence_length)
            for j in range(sequence_length):
                token_set.add(words_list[i][j])
        for token in list(token_set):
            self.add_token(token)

    def update(self, words):
        sequence_length = len(words) + 2  # for bos, ens token
        self.max_sequence_length = max(sequence_length, self.max_sequence_length)
        for word in words:
            self.add_token(word)

    def pad_sequence(self, sequence):
        sequence_padded = np.zeros(shape=(self.max_sequence_length,), dtype=np.int32)
        sequence_length = min(len(sequence), self.max_sequence_length)
        sequence_padded[:sequence_length] = sequence[:sequence_length]
        return sequence_padded

    def text_to_sequence(self, text):
        min_length = min(len(text), self.max_sequence_length)
        sequence = []
        for i in range(min_length):
            try:
                index = self.word_index[text[i]]
            except:
                index = self.word_index[self.unk_token]
            sequence.append(index)
        sequence = np.asarray(sequence).astype(np.int32)
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequences.append(self.text_to_sequence(text))
        return sequences

    def sequence_to_text(self, sequence):
        words = []
        for i in range(len(sequence)):
            if sequence[i] == self.bos_token:
                continue
            elif sequence[i] in [self.index_word[self.pad_token], self.index_word[self.end_token]]:
                break
            try:
                word = self.index_word[sequence[i]]
            except:
                word = self.unk_token
            words.append(word)
        text = ' '.join(words)
        return text

    def get_bos_sequence(self):
        sequence = np.zeros(shape=(self.max_sequence_length,), dtype=np.int32)
        sequence[0] = self.word_index[self.bos_token]
        return sequence


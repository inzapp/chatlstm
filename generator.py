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
import json
import konlpy
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from tokenizer import Tokenizer
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 data_path,
                 batch_size,
                 pretrained_model_output_size=0,
                 pretrained_vocab_size=0,
                 training=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.training = training
        self.pretrained_model_output_size = pretrained_model_output_size
        self.pretrained_vocab_size = pretrained_vocab_size
        self.morph_analyzer = konlpy.tag.Hannanum()
        self.tokenizer = Tokenizer()
        self.pool = ThreadPoolExecutor(8)
        self.pad_token_index = None
        self.bos_token_index = None
        self.eos_token_index = None
        self.utterance_sequences_list = None
        self.prepared = False
        self.json_paths = self.get_json_paths(self.data_path)
        self.json_index = 0
        # np.random.shuffle(self.json_paths)
        assert len(self.json_paths) > 0, f'json data not found : {self.data_path}'

    def get_json_paths(self, data_path):
        return glob(f'{data_path}/**/*.json', recursive=True)

    def load_json(self, json_path):
        with open(json_path, mode='rt', encoding='utf-8') as f:
            d = json.load(f)
        return d

    def split_words(self, nl):
        return self.morph_analyzer.morphs(nl)
        # return nl.split()

    def is_file_valid(self, path):
        return os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0

    def prepare(self, load_utterances=True):
        if self.prepared:
            return

        cache_file_path = f'{self.data_path}/data.cache'
        if not load_utterances:
            if self.is_file_valid(cache_file_path):
                self.tokenizer.load(cache_file_path)
            else:
                load_utterances = True

        if load_utterances:
            fs = []
            for path in self.json_paths:
                fs.append(self.pool.submit(self.load_json, path))
            self.utterance_sequences_list = []
            for f in tqdm(fs):
                d = f.result()
                utterances = d['utterances']
                if len(utterances) > 1:
                    utterance_length = len(utterances) if len(utterances) % 2 == 0 else len(utterances) - 1
                    utterance_sequences = []
                    for i in range(utterance_length):
                        nl = utterances[i]['text']
                        words = self.preprocess(nl, target='words')
                        self.tokenizer.update(words)
                        sequence = self.tokenizer.text_to_sequence(words)
                        utterance_sequences.append(sequence)
                    self.utterance_sequences_list.append(utterance_sequences)
            self.tokenizer.save(cache_file_path)

        data_vocab_size = self.tokenizer.vocab_size
        data_model_output_size = data_vocab_size + 2
        data_max_sequence_length = self.tokenizer.max_sequence_length
        if self.training:
            if self.pretrained_vocab_size > 0 or self.pretrained_model_output_size > 0:
                msg = f'pretrained_vocab_size({self.pretrained_vocab_size}) must be equal to data_vocab_size({data_vocab_size})'
                assert self.pretrained_vocab_size == data_vocab_size, msg
                msg = f'pretrained_model_output_size({self.pretrained_model_output_size}) must be equal to data_model_output_size({data_model_output_size})'
                assert self.pretrained_model_output_size == data_max_sequence_length, msg

        self.pad_token_index = self.tokenizer.word_index[self.tokenizer.pad_token]
        self.bos_token_index = self.tokenizer.word_index[self.tokenizer.bos_token]
        self.eos_token_index = self.tokenizer.word_index[self.tokenizer.eos_token]
        self.prepared = True

    def load(self, evaluate_bos=False):
        assert self.prepared
        encoder_batch_x, decoder_batch_x, batch_y = [], [], []
        utterance_sequences_indices = np.random.choice(len(self.utterance_sequences_list), self.batch_size, replace=False)
        for i in utterance_sequences_indices:
            utterance_sequences = self.utterance_sequences_list[i]
            utterance_sequences_index = np.random.randint(len(utterance_sequences) // 2)
            x_sequence = utterance_sequences[utterance_sequences_index]
            y_sequence = utterance_sequences[utterance_sequences_index+1]
            encoder_batch_x.append(self.tokenizer.pad_sequence(x_sequence))
            if self.training:
                random_index = np.random.randint(len(y_sequence) + 1)
            else:
                if evaluate_bos:
                    random_index = 0
                else:
                    random_index = np.random.randint(len(y_sequence)) + 1
            bos_sequence = np.asarray([self.bos_token_index])
            if random_index == 0:
                decoder_x = bos_sequence
                y = y_sequence[0]
            elif random_index == len(y_sequence):
                decoder_x = np.concatenate([bos_sequence, y_sequence])
                y = self.eos_token_index
            else:
                decoder_x = np.concatenate([bos_sequence, np.asarray(y_sequence[:random_index])])
                y = y_sequence[random_index]
            decoder_batch_x.append(self.tokenizer.pad_sequence(decoder_x))
            batch_y.append(y)
        encoder_batch_x = np.asarray(encoder_batch_x).astype(np.int32)
        decoder_batch_x = np.asarray(decoder_batch_x).astype(np.int32)
        batch_y = np.asarray(batch_y).astype(np.int32)
        return [encoder_batch_x, decoder_batch_x], batch_y

    # def next_json_path(self):
    #     json_path = self.json_paths[self.json_index]
    #     self.json_index += 1
    #     if self.json_index == len(self.json_paths):
    #         self.json_index = 0
    #         np.random.shuffle(self.json_paths)
    #     return json_path

    def evaluate_generator(self, evaluate_bos):
        assert self.prepared
        for _ in range(len(self.utterance_sequences_list) // self.batch_size):
            yield self.load(evaluate_bos=evaluate_bos)

    def preprocess(self, nl, target):
        assert target in ['words', 'sequence', 'sequence_pad']
        nl = nl.strip()
        nl = self.tokenizer.remove_special_tokens(nl)
        words = self.split_words(nl)
        if target == 'words':
            return words
        sequence = self.tokenizer.text_to_sequence(words)
        if target == 'sequence':
            x = np.asarray(sequence).astype(np.int32)
        else:
            sequence_padded = self.tokenizer.pad_sequence(sequence)
            x = np.asarray(sequence_padded).astype(np.int32)
        return x

    def postprocess(self, y):
        assert self.prepared
        index = np.argmax(y)
        end = index in [self.pad_token_index, self.eos_token_index]
        if end:
            return None, None, end
        else:
            return index, self.tokenizer.index_word[index], end


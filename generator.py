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
        self.tokenizer = Tokenizer()
        self.pool = ThreadPoolExecutor(8)
        self.ds = None
        self.prepared = False
        self.json_paths = self.get_json_paths(self.data_path)
        self.json_index = 0
        np.random.shuffle(self.json_paths)
        assert len(self.json_paths) > 0, f'json data not found : {self.data_path}'

    def get_json_paths(self, data_path):
        return glob(f'{data_path}/**/*.json', recursive=True)

    def load_json(self, json_path):
        with open(json_path, mode='rt', encoding='utf-8') as f:
            d = json.load(f)
        return d, json_path

    def get_json_value(self, d, key):
        value = None
        try:
            value = d[key]
        except:
            pass
        return value

    def is_file_valid(self, path):
        return os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0

    def prepare(self, load_data=True):
        if self.prepared:
            return

        cache_file_path = f'{self.data_path}/data.cache'
        if not load_data:
            if self.is_file_valid(cache_file_path):
                self.tokenizer.load(cache_file_path)
            else:
                load_data = True

        if load_data:
            fs = []
            for path in self.json_paths:
                fs.append(self.pool.submit(self.load_json, path))
            self.ds = []
            for f in tqdm(fs):
                d, path = f.result()
                data_type = self.get_json_value(d, 'type')
                if data_type is None:
                    print(f'json key "type" is not found : {path}')
                    continue

                if data_type == 'text':
                    nl = self.get_json_value(d, 'content')
                    if nl is None:
                        print(f'json key "content" is not found : {path}')
                        continue

                    self.tokenizer.update(nl)
                    self.ds.append(d)
                elif data_type == 'dialogue':
                    dialogues = self.get_json_value(d, 'content')
                    if dialogues is None:
                        print(f'json key "content" is not found : {path}')
                        continue

                    error = False
                    input_nls = []
                    output_nls = []
                    for dialogue in dialogues:
                        input_nl = self.get_json_value(dialogue, 'input')
                        if input_nl is None:
                            print(f'json key "input" is not found : {path}')
                            error = True
                            break

                        output_nl = self.get_json_value(dialogue, 'output')
                        if output_nl is None:
                            print(f'json key "output" is not found : {path}')
                            error = True
                            break

                        if input_nl is not None and output_nl is not None:
                            input_nls.append(input_nl)
                            output_nls.append(output_nl)

                    if not error:
                        for input_nl in input_nls:
                            self.tokenizer.update(input_nl)
                        for output_nl in output_nls:
                            self.tokenizer.update(output_nl)
                        self.ds.append(d)

                else:
                    print(f'invalid data_type "{data_type}" : {path}')
            self.tokenizer.save(cache_file_path)

        data_vocab_size = self.tokenizer.vocab_size
        data_model_output_size = data_vocab_size + self.tokenizer.bos_eos_token_margin
        data_max_sequence_length = self.tokenizer.max_sequence_length
        if self.training:
            if self.pretrained_vocab_size > 0 or self.pretrained_model_output_size > 0:
                msg = f'pretrained_vocab_size({self.pretrained_vocab_size}) must be equal to data_vocab_size({data_vocab_size})'
                assert self.pretrained_vocab_size == data_vocab_size, msg
                msg = f'pretrained_model_output_size({self.pretrained_model_output_size}) must be equal to data_model_output_size({data_model_output_size})'
                assert self.pretrained_model_output_size == data_max_sequence_length, msg
        self.prepared = True

    def load(self):
        assert self.prepared
        batch_x, batch_y = [], []
        batch_indices = np.random.choice(len(self.ds), self.batch_size, replace=False)
        for i in batch_indices:
            d = self.ds[i]
            if d['type'] == 'text':
                nl = d['content']
                sequence = self.preprocess(nl, target='sequence', data_type='text')
                random_index = np.random.randint(len(sequence))
                if random_index == 0:
                    x_sequence = sequence[:1]
                    y = sequence[1]
                elif random_index == len(sequence) - 1:
                    x_sequence = sequence
                    y = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
                else:
                    x_sequence = sequence[:random_index]
                    y = sequence[random_index]
            elif d['type'] == 'dialogue':
                dialogues = d['content']
                dialogue_index = np.random.randint(len(dialogues))
                dialogue = dialogues[dialogue_index]
                input_nl = dialogue['input']
                output_nl = dialogue['output']
                input_sequence = self.preprocess(input_nl, target='sequence', data_type='dialogue_input')
                output_sequence = self.preprocess(output_nl, target='sequence', data_type='dialogue_output')
                random_index = np.random.randint(len(output_sequence))
                if random_index == 0:
                    x_sequence = np.append(input_sequence, output_sequence[1:])
                    y = output_sequence[1]
                elif random_index == len(output_sequence) - 1:
                    x_sequence = np.append(input_sequence, output_sequence)
                    y = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
                else:
                    x_sequence = np.append(input_sequence, output_sequence[:random_index])
                    y = output_sequence[random_index]
            batch_x.append(self.tokenizer.convert_sequence_to_padded_sequence(x_sequence, unk_dropout=0.05 if self.training else 0.0))
            batch_y.append(y)
        batch_x = np.asarray(batch_x).astype(np.int32)
        batch_y = np.asarray(batch_y).astype(np.int32)
        return batch_x, batch_y

    # def next_json_path(self):
    #     json_path = self.json_paths[self.json_index]
    #     self.json_index += 1
    #     if self.json_index == len(self.json_paths):
    #         self.json_index = 0
    #         np.random.shuffle(self.json_paths)
    #     return json_path

    def evaluate_generator(self):
        assert self.prepared
        for _ in range(len(self.ds) // self.batch_size):
            yield self.load()

    def preprocess(self, nl, target, data_type):
        assert target in ['tokens', 'sequence', 'padded_sequence']
        assert data_type in ['text', 'dialogue_input', 'dialogue_output']
        nl = self.tokenizer.preprocess(nl)
        tokens = self.tokenizer.convert_nl_to_tokens(nl)
        if target == 'tokens':
            return tokens

        sequence = self.tokenizer.convert_tokens_to_sequence(tokens)

        if data_type == 'dialogue_input':
            sequence = np.append(self.tokenizer.token_to_index_dict[self.tokenizer.bos_user_token], sequence)
        elif data_type == 'dialogue_output':
            sequence = np.append(self.tokenizer.token_to_index_dict[self.tokenizer.bos_assistant_token], sequence)

        if target == 'sequence':
            return sequence

        padded_sequence = self.tokenizer.convert_sequence_to_padded_sequence(sequence)
        if target == 'padded_sequence':
            return padded_sequence

        print(f'invalid preprocess target : {target}')

    def postprocess(self, y):
        assert self.prepared
        index = np.argmax(y)
        pad_token_index = self.tokenizer.token_to_index_dict[self.tokenizer.pad_token]
        eos_token_index = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
        end = index in [eos_token_index, pad_token_index]
        if end:
            return None, None, end
        else:
            return index, self.tokenizer.index_to_token_dict[index], end


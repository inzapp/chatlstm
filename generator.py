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
import sys
import json
import signal
import threading
import numpy as np

from glob import glob
from tqdm import tqdm
from time import sleep
from collections import deque
from tokenizer import Tokenizer
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, cfg, data_path, pretrained_model_output_size=0, pretrained_vocab_size=0, training=False):
        self.cfg = cfg
        self.data_path = data_path
        self.training = training
        self.pretrained_model_output_size = pretrained_model_output_size
        self.pretrained_vocab_size = pretrained_vocab_size
        self.tokenizer = Tokenizer()
        self.pool = ThreadPoolExecutor(8)
        self.json_paths = self.get_json_paths(self.data_path)
        self.json_index = 0
        np.random.shuffle(self.json_paths)
        assert len(self.json_paths) > 0, f'json data not found : {self.data_path}'

        self.lock = threading.Lock()
        self.q_thread = threading.Thread(target=self.load_xy_into_q)
        self.q_thread.daemon = True
        self.q = deque()
        self.q_thread_running = False
        self.q_thread_pause = False
        self.q_indices = list(range(self.cfg.max_q_size))

    def signal_handler(self, sig, frame):
        print()
        print(f'{signal.Signals(sig).name} signal detected, please wait until the end of the thread')
        self.stop()
        print(f'exit successfully')
        sys.exit(0)

    def start(self):
        self.q_thread_running = True
        self.q_thread.start()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while True:
            sleep(1.0)
            percentage = (len(self.q) / self.cfg.max_q_size) * 100.0
            print(f'prefetching training data... {percentage:.1f}%')
            with self.lock:
                if len(self.q) >= self.cfg.max_q_size:
                    print()
                    break

    def stop(self):
        if self.q_thread_running:
            self.q_thread_running = False
            while self.q_thread.is_alive():
                sleep(0.1)

    def pause(self):
        if self.q_thread_running:
            self.q_thread_pause = True

    def resume(self):
        if self.q_thread_running:
            self.q_thread_pause = False

    def exit(self):
        self.signal_handler(signal.SIGINT, None)

    def is_json_data_valid(self, d, path, verbose=False):
        data_type = self.get_json_value(d, 'type')
        if data_type is None:
            if verbose:
                print(f'json key "type" is not found : {path}')
            return False

        if data_type not in ['text', 'dialogue']:
            if verbose:
                print(f'invalid data type {data_type} : path')
            return False

        if data_type == 'text':
            nl = self.get_json_value(d, 'content')
            if nl is None:
                if verbose:
                    print(f'json key "content" is not found : {path}')
                return False

            if type(nl) is not str:
                if verbose:
                    print(f'content type is not str : {path}')
                return False

            if len(nl) == 0:
                if verbose:
                    print(f'content nl length is zero : {path}')
                return False
        elif data_type == 'dialogue':
            dialogues = self.get_json_value(d, 'content')
            if dialogues is None:
                if verbose:
                    print(f'json key "content" is not found : {path}')
                return False

            if type(dialogues) is not list:
                if verbose:
                    print(f'content type is not list : {path}')
                return False

            if len(dialogues) == 0:
                if verbose:
                    print(f'content dialogue list size is zero : {path}')
                return False

            for dialogue in dialogues:
                input_nl = self.get_json_value(dialogue, 'input')
                if input_nl is None:
                    if verbose:
                        print(f'json key "input" is not found : {path}')
                    return False

                if type(input_nl) is not str:
                    if verbose:
                        print(f'input_nl type is not string : {path}')
                    return False

                if len(input_nl) == 0:
                    if verbose:
                        print(f'input_nl length is zero : {path}')
                    return False

                output_nl = self.get_json_value(dialogue, 'output')
                if output_nl is None:
                    if verbose:
                        print(f'json key "output" is not found : {path}')
                    return False

                if type(output_nl) is not str:
                    if verbose:
                        print(f'output_nl type is not string : {path}')
                    return False

                if len(output_nl) == 0:
                    if verbose:
                        print(f'output_nl length is zero : {path}')
                    return False
        return True

    def load_tokenizer(self):
        if self.tokenizer.is_loaded:
            return
        else:
            if self.cfg.pretrained_tokenizer_path is not None and self.is_file_valid(self.cfg.pretrained_tokenizer_path):
                self.tokenizer.load(self.cfg.pretrained_tokenizer_path)
                return

            data_tokenizer_file_path = f'{self.data_path}/{Tokenizer.default_file_name}'
            if self.is_file_valid(data_tokenizer_file_path):
                self.tokenizer.load(data_tokenizer_file_path)
                return

            fs = []
            for path in self.json_paths:
                fs.append(self.pool.submit(self.load_json, path))
            for f in tqdm(fs):
                d, path = f.result()
                if self.is_json_data_valid(d, path, verbose=True):
                    data_type = d['type']
                    if data_type == 'text':
                        nl = d['content']
                        self.tokenizer.update(nl)
                    elif data_type == 'dialogue':
                        dialogues = d['content']
                        for dialogue in dialogues:
                            input_nl = dialogue['input']
                            output_nl = dialogue['output']
                            self.tokenizer.update(input_nl)
                            self.tokenizer.update(output_nl)
            self.tokenizer.save(data_tokenizer_file_path)

    def load_xy(self):
        json_path = self.next_json_path()
        d, path = self.load_json(json_path)
        if not self.is_json_data_valid(d, path):
            return None, None

        x_sequence, y_index = None, None
        if d['type'] == 'text':
            nl = d['content']
            sequence = self.preprocess(nl, target='sequence', data_type='text')
            random_index = np.random.randint(len(sequence))
            if random_index == len(sequence) - 1:
                x_sequence = sequence
                y_index = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
            elif random_index == 0:
                x_sequence = sequence[:1]
                y_index = sequence[1]
            else:
                x_sequence = sequence[:random_index]
                y_index = sequence[random_index]
        elif d['type'] == 'dialogue':
            dialogues = d['content']
            dialogue_index = np.random.randint(len(dialogues))
            input_sequence = np.array([], dtype=np.int32)
            output_sequence = np.array([], dtype=np.int32)
            for i in range(max(dialogue_index - 2, 0), dialogue_index + 1, 1):
                dialogue = dialogues[i]
                cur_input_nl = dialogue['input']
                cur_output_nl = dialogue['output']
                cur_input_sequence = self.preprocess(cur_input_nl, target='sequence', data_type='dialogue_input')
                cur_output_sequence = self.preprocess(cur_output_nl, target='sequence', data_type='dialogue_output')
                if i < dialogue_index:
                    input_sequence = np.append(input_sequence, cur_input_sequence)
                    input_sequence = np.append(input_sequence, cur_output_sequence)
                else:
                    input_sequence = np.append(input_sequence, cur_input_sequence)
                    random_index = np.random.randint(len(cur_output_sequence))
                    if random_index == len(cur_output_sequence) - 1:
                        x_sequence = np.append(input_sequence, cur_output_sequence)
                        y_index = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
                    elif random_index == 0:
                        x_sequence = np.append(input_sequence, cur_output_sequence[1:])
                        y_index = cur_output_sequence[1]
                    else:
                        x_sequence = np.append(input_sequence, cur_output_sequence[:random_index])
                        y_index = cur_output_sequence[random_index]

        x, y = None, None
        if x_sequence is not None and y_index is not None:
            x = self.tokenizer.convert_sequence_to_padded_sequence(x_sequence, unk_dropout=self.cfg.unk_dropout if self.training else 0.0)
            x = np.asarray(x).reshape((self.tokenizer.max_sequence_length,)).astype(np.int32)
            y = int(y_index)
        return x, y

    def load_xy_into_q(self):
        while self.q_thread_running:
            if self.q_thread_pause:
                sleep(1.0)
            else:
                x, y = self.load_xy()
                if x is not None and y is not None:
                    with self.lock:
                        if len(self.q) == self.cfg.max_q_size:
                            self.q.popleft()
                        self.q.append((x, y))

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

    # def prepare(self, load_data=True):
        # self.load_tokenizer()
        # if self.prepared:
        #     return

        # tokenizer_file_path = f'{self.data_path}/tokenizer.data'
        # if not load_data:
        #     if self.is_file_valid(tokenizer_file_path):
        #         self.tokenizer.load(tokenizer_file_path)
        #     else:
        #         load_data = True

        # if load_data:
        #     fs = []
        #     for path in self.json_paths:
        #         fs.append(self.pool.submit(self.load_json, path))
        #     self.ds = []
        #     for f in tqdm(fs):
        #         d, path = f.result()
        #         data_type = self.get_json_value(d, 'type')
        #         if data_type is None:
        #             print(f'json key "type" is not found : {path}')
        #             continue

        #         if data_type == 'text':
        #             nl = self.get_json_value(d, 'content')
        #             if nl is None:
        #                 print(f'json key "content" is not found : {path}')
        #                 continue

        #             self.tokenizer.update(nl)
        #             self.ds.append(d)
        #         elif data_type == 'dialogue':
        #             dialogues = self.get_json_value(d, 'content')
        #             if dialogues is None:
        #                 print(f'json key "content" is not found : {path}')
        #                 continue

        #             error = False
        #             input_nls = []
        #             output_nls = []
        #             for dialogue in dialogues:
        #                 input_nl = self.get_json_value(dialogue, 'input')
        #                 if input_nl is None:
        #                     print(f'json key "input" is not found : {path}')
        #                     error = True
        #                     break

        #                 output_nl = self.get_json_value(dialogue, 'output')
        #                 if output_nl is None:
        #                     print(f'json key "output" is not found : {path}')
        #                     error = True
        #                     break

        #                 if input_nl is not None and output_nl is not None:
        #                     input_nls.append(input_nl)
        #                     output_nls.append(output_nl)

        #             if not error:
        #                 for input_nl in input_nls:
        #                     self.tokenizer.update(input_nl)
        #                 for output_nl in output_nls:
        #                     self.tokenizer.update(output_nl)
        #                 self.ds.append(d)

        #         else:
        #             print(f'invalid data_type "{data_type}" : {path}')
        #     self.tokenizer.save(tokenizer_file_path)

        # data_vocab_size = self.tokenizer.vocab_size
        # data_model_output_size = data_vocab_size + self.tokenizer.bos_eos_token_margin
        # data_max_sequence_length = self.tokenizer.max_sequence_length
        # if self.training:
        #     if self.pretrained_vocab_size > 0 or self.pretrained_model_output_size > 0:
        #         msg = f'pretrained_vocab_size({self.pretrained_vocab_size}) must be equal to data_vocab_size({data_vocab_size})'
        #         assert self.pretrained_vocab_size == data_vocab_size, msg
        #         msg = f'pretrained_model_output_size({self.pretrained_model_output_size}) must be equal to data_model_output_size({data_model_output_size})'
        #         assert self.pretrained_model_output_size == data_max_sequence_length, msg
        # self.prepared = True

    def load(self):
        batch_x, batch_y = [], []
        for i in np.random.choice(self.q_indices, self.cfg.batch_size, replace=False):
            with self.lock:
                x, y = self.q[i]
                batch_x.append(x)
                batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.cfg.batch_size, self.tokenizer.max_sequence_length)).astype(np.int32)
        batch_y = np.asarray(batch_y).reshape((self.cfg.batch_size, 1)).astype(np.int32)
        return batch_x, batch_y

    def next_json_path(self):
        json_path = self.json_paths[self.json_index]
        self.json_index += 1
        if self.json_index == len(self.json_paths):
            self.json_index = 0
            np.random.shuffle(self.json_paths)
        return json_path

    def evaluate_generator(self):
        assert self.tokenizer.is_loaded
        for _ in range(len(self.json_paths) // self.cfg.batch_size):
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
        assert self.tokenizer.is_loaded
        index = np.argmax(y)
        pad_token_index = self.tokenizer.token_to_index_dict[self.tokenizer.pad_token]
        eos_token_index = self.tokenizer.token_to_index_dict[self.tokenizer.eos_token]
        end = index in [eos_token_index, pad_token_index]
        if end:
            return None, None, end
        else:
            return index, self.tokenizer.index_to_token_dict[index], end


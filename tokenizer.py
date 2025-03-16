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
import json
import konlpy
import numpy as np


class Tokenizer:
    def __init__(self):
        self.index_to_token_dict = {}
        self.token_to_index_dict = {}
        self.vocab_size = 0
        self.max_sequence_length = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_user_token = '<BOS_USER>'
        self.bos_assistant_token = '<BOS_ASSISTANT>'
        self.eos_token = '<EOS>'
        self.bos_tokens = [self.bos_user_token, self.bos_assistant_token]
        self.special_tokens = [self.pad_token, self.unk_token] + self.bos_tokens + [self.eos_token] + ['\n']
        self.morph_analyzer = konlpy.tag.Hannanum()
        self.bos_eos_token_margin = 16
        self.init()

    def remove_special_tokens(self, nl):
        updated_count = 0
        for special_token in self.special_tokens:
            if nl.find(special_token) > -1:
                nl = nl.replace(special_token, '')
                updated_count += 1
        return nl, updated_count

    def replace_crlf_and_cr_to_lf(self, nl):
        updated_count = 0
        cr_chars = ['\r\n', '\r']
        for cr_char in cr_chars:
            if nl.find(cr_char) > -1:
                nl = nl.replace(cr_char, '\n')
                updated_count += 1
        return nl, updated_count

    def remove_duplicate_spaces(self, nl):
        updated_count = 0
        if nl.find('  ') > -1:
            nl = nl.replace('  ', ' ')
            updated_count += 1
        return nl, updated_count

    def remove_duplicate_lf(self, nl):
        updated_count = 0
        if nl.find('\n\n\n') > -1:
            nl = nl.replace('\n\n\n', '\n\n')
            updated_count += 1
        return nl, updated_count

    def preprocess(self, nl):
        nl = nl.strip()
        while True:
            uc_sum = 0
            nl, uc = self.remove_special_tokens(nl)
            uc_sum += uc
            nl, uc = self.replace_crlf_and_cr_to_lf(nl)
            uc_sum += uc
            nl, uc = self.remove_duplicate_spaces(nl)
            uc_sum += uc
            nl, uc = self.remove_duplicate_lf(nl)
            uc_sum += uc
            if uc_sum == 0:
                break
        return nl

    def add_token(self, token):
        if not (token in self.token_to_index_dict):
            self.index_to_token_dict[self.vocab_size] = token
            self.token_to_index_dict[token] = self.vocab_size
            self.vocab_size += 1

    def init(self):
        self.index_to_token_dict = {}
        self.token_to_index_dict = {}
        self.vocab_size = 0
        self.max_sequence_length = 0
        for special_token in self.special_tokens:
            self.add_token(special_token)

    def update(self, nl):
        nl = self.preprocess(nl)
        tokens = self.convert_nl_to_tokens(nl)
        sequence_length = len(tokens) + self.bos_eos_token_margin
        self.max_sequence_length = max(sequence_length, self.max_sequence_length)
        for token in tokens:
            self.add_token(token)

    def convert_nl_to_tokens(self, nl):
        return self.morph_analyzer.morphs(nl)

    def convert_tokens_to_sequence(self, tokens):
        min_length = min(len(tokens), self.max_sequence_length)
        sequence = []
        for i in range(min_length):
            try:
                index = self.token_to_index_dict[tokens[i]]
            except:
                index = self.token_to_index_dict[self.unk_token]
            sequence.append(index)
        sequence = np.asarray(sequence).astype(np.int32)
        return sequence

    def convert_sequence_to_padded_sequence(self, sequence, unk_dropout=0.0):
        assert 0.0 <= unk_dropout <= 1.0
        sequence_padded = np.zeros(shape=(self.max_sequence_length,), dtype=np.int32)
        sequence_length = min(len(sequence), self.max_sequence_length)
        sequence_padded[:sequence_length] = sequence[:sequence_length]
        if unk_dropout > 0.0:
            unk_ratio = np.random.uniform() * unk_dropout
            unk_count = int(unk_ratio * sequence_length)
            unk_indices = np.random.choice(sequence_length, unk_count, replace=False)
            sequence_padded[unk_indices] = self.token_to_index_dict[self.unk_token]
        return sequence_padded

    def convert_sequence_to_nl(self, sequence):
        tokens = []
        for token_index in sequence:
            if token_index in [self.token_to_index_dict[end_token] for end_token in [self.eos_token, self.pad_token]]:
                break
            if token_index in [self.token_to_index_dict[bos_token] for bos_token in self.bos_tokens]:
                continue
            try:
                token = self.index_to_token_dict[token_index]
            except:
                token = self.unk_token
            tokens.append(token)
        nl = ' '.join(tokens)
        return nl

    def save(self, path):
        d = {}
        d['vocab_size'] = self.vocab_size
        d['max_sequence_length'] = self.max_sequence_length
        d['index_to_token_dict'] = self.index_to_token_dict
        d['token_to_index_dict'] = self.token_to_index_dict
        with open(path, mode='wt', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=True, indent=4)

    def convert_keys_to_int(self, d):
        new_dict = {}
        for k, v in d.items():
            new_dict[int(k)] = v
        return new_dict

    def convert_values_to_int(self, d):
        new_dict = {}
        for k, v in d.items():
            new_dict[k] = int(v)
        return new_dict

    def load(self, path):
        with open(path, mode='rt', encoding='utf-8') as f:
            d = json.load(f)
        self.vocab_size = int(d['vocab_size'])
        self.max_sequence_length = int(d['max_sequence_length'])
        self.index_to_token_dict = d['index_to_token_dict']
        self.token_to_index_dict = d['token_to_index_dict']
        self.index_to_token_dict = self.convert_keys_to_int(self.index_to_token_dict)
        self.token_to_index_dict = self.convert_values_to_int(self.token_to_index_dict)


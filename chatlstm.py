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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import yaml
import random
import numpy as np
import shutil as sh

from model import Model
from eta import ETACalculator
from tokenizer import Tokenizer
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self, cfg_path):
        self.__d = self.load(cfg_path)
        self.sync_attribute()

    def sync_attribute(self):
        for key, value in self.__d.items():
            setattr(self, key, value)

    def __get_value_from_yaml(self, cfg, key, default, parse_type, required):
        try:
            value = parse_type(cfg[key])
            if parse_type is str and value.lower() in ['none', 'null']:
                value = None
            return value
        except:
            if required:
                Logger.error(f'cfg parse failure, {key} is required')
            return default

    def set_config(self, key, value):
        self.__d[key] = value
        setattr(self, key, value)

    def load(self, cfg_path):
        cfg = None
        if not (os.path.exists(cfg_path) and os.path.isfile(cfg_path)):
            Logger.error(f'cfg not found, path : {cfg_path}')

        with open(cfg_path, 'rt') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        d = {}
        d['devices'] = self.__get_value_from_yaml(cfg, 'devices', [0], list, required=False)
        d['pretrained_model_path'] = self.__get_value_from_yaml(cfg, 'pretrained_model_path', None, str, required=False)
        d['pretrained_tokenizer_path'] = self.__get_value_from_yaml(cfg, 'pretrained_tokenizer_path', None, str, required=False)
        d['train_data_path'] = self.__get_value_from_yaml(cfg, 'train_data_path', None, str, required=True)
        d['validation_data_path'] = self.__get_value_from_yaml(cfg, 'validation_data_path', None, str, required=True)
        d['model_name'] = self.__get_value_from_yaml(cfg, 'model_name', 'model', str, required=False)
        d['optimizer'] = self.__get_value_from_yaml(cfg, 'optimizer', 'adam', str, required=False)
        d['lr_policy'] = self.__get_value_from_yaml(cfg, 'lr_policy', 'step', str, required=False)
        d['lr'] = self.__get_value_from_yaml(cfg, 'lr', 0.001, float, required=False)
        d['lrf'] = self.__get_value_from_yaml(cfg, 'lrf', 0.05, float, required=False)
        d['l2'] = self.__get_value_from_yaml(cfg, 'l2', 0.0005, float, required=False)
        d['dropout'] = self.__get_value_from_yaml(cfg, 'dropout', 0.0, float, required=False)
        d['unk_dropout'] = self.__get_value_from_yaml(cfg, 'unk_dropout', 0.0, float, required=False)
        warm_up = self.__get_value_from_yaml(cfg, 'warm_up', 1000, float, required=False)
        d['warm_up'] = float(warm_up) if 0.0 <= warm_up <= 1.0 else int(warm_up)
        d['momentum'] = self.__get_value_from_yaml(cfg, 'momentum', 0.9, float, required=False)
        d['smoothing'] = self.__get_value_from_yaml(cfg, 'smoothing', 0.0, float, required=False)
        d['embedding_dim'] = self.__get_value_from_yaml(cfg, 'embedding_dim', 128, int, required=False)
        d['max_conv_filters'] = self.__get_value_from_yaml(cfg, 'max_conv_filters', 4096, int, required=False)
        d['recurrent_units'] = self.__get_value_from_yaml(cfg, 'recurrent_units', 256, int, required=False)
        d['batch_size'] = self.__get_value_from_yaml(cfg, 'batch_size', 4, int, required=False)
        d['max_q_size'] = self.__get_value_from_yaml(cfg, 'max_q_size', 1024, int, required=False)
        d['iterations'] = self.__get_value_from_yaml(cfg, 'iterations', None, int, required=True)
        d['checkpoint_interval'] = self.__get_value_from_yaml(cfg, 'checkpoint_interval', 0, int, required=False)
        d['use_gru'] = self.__get_value_from_yaml(cfg, 'use_gru', False, bool, required=False)
        d['fix_seed'] = self.__get_value_from_yaml(cfg, 'fix_seed', False, bool, required=False)
        return d

    def save(self, cfg_path):
        with open(cfg_path, 'wt') as f:
            yaml.dump(self.__d, f, default_flow_style=False, sort_keys=False)

    def print_cfg(self):
        print(self.__d)


class ChatLSTM(CheckpointManager):
    def __init__(self, cfg, evaluate=False):
        super().__init__()
        assert cfg.max_q_size >= cfg.batch_size
        self.cfg = cfg

        if self.cfg.checkpoint_interval == 0:
            self.cfg.checkpoint_interval = self.cfg.iterations

        if evaluate or self.cfg.fix_seed:
            self.set_global_seed()

        self.pretrained_iteration_count = 0

        self.model = None
        self.pretrained_model_output_size = 0
        self.pretrained_vocab_size = 0
        if self.cfg.pretrained_model_path is not None:
            if self.is_model_path_exists(self.cfg.pretrained_model_path):
                self.model = tf.keras.models.load_model(self.cfg.pretrained_model_path, compile=False)
                self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.cfg.pretrained_model_path)
                self.pretrained_model_output_size, self.pretrained_vocab_size = self.get_output_size_and_vocab_size_from_model(self.model)
                print(f'load pretrained model success : {self.cfg.pretrained_model_path}')
            else:
                print(f'pretrained model not found : {self.cfg.pretrained_model_path}')
                exit(0)

        self.train_data_generator = DataGenerator(
            cfg=self.cfg,
            data_path=self.cfg.train_data_path,
            pretrained_model_output_size=self.pretrained_model_output_size,
            pretrained_vocab_size=self.pretrained_vocab_size,
            training=True)
        self.validation_data_generator = DataGenerator(
            cfg=self.cfg,
            data_path=self.cfg.validation_data_path,
            pretrained_model_output_size=self.pretrained_model_output_size,
            pretrained_vocab_size=self.pretrained_vocab_size)

        if self.cfg.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr, beta_1=self.cfg.momentum)
        elif self.cfg.optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.cfg.lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def is_model_path_exists(self, model_path):
        return (model_path is not None) and os.path.exists(model_path) and os.path.isfile(model_path) and model_path.endswith('.h5')

    def compute_gradient(self, model, optimizer, loss_fn, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    def predict(self, model, nl, data_type):
        assert data_type in ['text', 'dialogue']
        if data_type == 'text':
            sequence = self.train_data_generator.preprocess(nl, target='sequence', data_type='text')
            x = self.train_data_generator.tokenizer.convert_sequence_to_padded_sequence(sequence)
        else:
            sequence = self.train_data_generator.preprocess(nl, target='sequence', data_type='dialogue_input')
            sequence = np.append(sequence, self.train_data_generator.tokenizer.token_to_index_dict[self.train_data_generator.tokenizer.bos_assistant_token])
            x = self.train_data_generator.tokenizer.convert_sequence_to_padded_sequence(sequence)
        x = np.asarray(x).reshape((1,) + x.shape)
        sequence_length = len(sequence)

        output_nl = ''
        max_length = self.train_data_generator.tokenizer.max_sequence_length - self.train_data_generator.tokenizer.bos_eos_token_margin - sequence_length
        for i in range(max_length):
            y = self.graph_forward(model, x)
            index, token, end = self.train_data_generator.postprocess(np.array(y[0]))
            if end:
                break
            if i == 0:
                output_nl = f'{token}'
            else:
                output_nl += f' {token}'
            x[0][sequence_length+i] = index
        return output_nl

    def compile(self, model):
        metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name=f'top_5_acc')
        model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[metric])

    def get_output_size_and_vocab_size_from_model(self, model):
        output_size = model.input_shape[-1]
        vocab_size = model.output_shape[-1]
        return output_size, vocab_size

    def chat(self, auto=False, auto_count=10, dataset='train'):
        assert dataset in ['train', 'validation']
        self.train_data_generator.load_tokenizer()
        if auto:
            data_generator = self.train_data_generator
            if dataset == 'validation':
                data_generator = self.validation_data_generator
                data_generator.tokenizer = self.train_data_generator.tokenizer
            print('chat start\n')
            i = 0
            valid_type_chat_count = 0
            while True:
                json_path = data_generator.json_paths[i]
                d, path = data_generator.load_json(json_path)
                if data_generator.is_json_data_valid(d, path) and d['type'] == 'dialogue':
                    if i > 0:
                        print()
                    dialogues = d['content']
                    dialogue = dialogues[0]
                    input_nl = dialogue['input']
                    output_nl = dialogue['output']
                    print(f'You : {input_nl}')
                    generated_nl = self.predict(self.model, input_nl, data_type='dialogue')
                    print(f'GT : {output_nl}')
                    print(f'AI : {generated_nl}')
                    valid_type_chat_count += 1
                    if valid_type_chat_count == auto_count:
                        break
                i += 1
                if i == len(data_generator.json_paths) or i > 300:
                    if i > 300:
                        print(f'dialogue type data not found in 300 json... maybe not exists?')
                    break
        else:
            print('chat start\n')
            while True:
                nl = input('You : ')
                if nl == 'q':
                    exit(0)
                output_nl = self.predict(self.model, nl, data_type='dialogue')
                print(f'AI : {output_nl}\n')

    def evaluate(self, dataset='train'):
        assert dataset in ['train', 'validation']
        self.train_data_generator.load_tokenizer()
        if dataset == 'train':
            data_generator = self.train_data_generator
        else:
            self.validation_data_generator.load_tokenizer()
            self.validation_data_generator.tokenizer = self.train_data_generator.tokenizer  # use trained tokenizer for validation
            data_generator = self.validation_data_generator
        self.compile(self.model)
        ret = self.model.evaluate(
            x=data_generator.evaluate_generator(),
            batch_size=self.cfg.batch_size,
            return_dict=True)
        acc = ret['top_5_acc']
        return acc

    def init_checkpoint_dir_extra(self):
        self.cfg.save(f'{self.checkpoint_path}/cfg.yaml')
        file_name = Tokenizer.default_file_name
        sh.copy(f'{self.cfg.train_data_path}/{file_name}', f'{self.checkpoint_path}/{file_name}')

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def train(self):
        self.train_data_generator.load_tokenizer()
        self.validation_data_generator.load_tokenizer()
        if self.model is None:
            data_max_sequence_length = self.train_data_generator.tokenizer.max_sequence_length
            data_vocab_size = self.train_data_generator.tokenizer.vocab_size
            self.model = Model(
                cfg=self.cfg,
                max_sequence_length=data_max_sequence_length,
                vocab_size=data_vocab_size).build()

        self.compile(self.model)
        self.model.summary()
        print()
        self.cfg.print_cfg()
        print()
        print(f'vocab size : {self.train_data_generator.tokenizer.vocab_size}')
        print(f'max sequence length : {self.train_data_generator.tokenizer.max_sequence_length}')
        print(f'train on {len(self.train_data_generator.json_paths)} samples\n')
        self.train_data_generator.start()
        if self.cfg.pretrained_model_path is not None:
            print(f'start training with pretrained model : {self.cfg.pretrained_model_path}')
        else:
            print('start training')

        self.init_checkpoint_dir(model_name=self.cfg.model_name, extra_function=self.init_checkpoint_dir_extra)
        iteration_count = self.pretrained_iteration_count
        compute_gradient = tf.function(self.compute_gradient)
        lr_scheduler = LRScheduler(lr=self.cfg.lr, lrf=self.cfg.lrf, iterations=self.cfg.iterations, warm_up=self.cfg.warm_up, policy=self.cfg.lr_policy)
        eta_calculator = ETACalculator(iterations=self.cfg.iterations)
        eta_calculator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(self.optimizer, iteration_count)
            loss = compute_gradient(self.model, self.optimizer, self.loss_fn, batch_x, batch_y)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            warm_up_end = iteration_count >= lr_scheduler.warm_up_iterations
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count)
            if warm_up_end:
                if iteration_count % self.cfg.checkpoint_interval == 0:
                    print()
                    acc = self.evaluate()
                    best_model_path = self.save_best_model(self.model, iteration_count, metric=acc, mode='max', content=f'_acc_{acc:.4f}')
                    if best_model_path:
                        print(f'[{iteration_count} iter] evaluation success with acc {acc:.4f}, new best model is saved to {best_model_path}\n')
                    else:
                        print(f'[{iteration_count} iter] evaluation success with acc {acc:.4f}\n')
            if iteration_count == self.cfg.iterations:
                # self.remove_last_model()
                self.train_data_generator.stop()
                print('train end successfully')
                return


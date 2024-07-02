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
import warnings
warnings.filterwarnings(action='ignore')
import random
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self,
                 train_data_path,
                 validation_data_path,
                 model_name,
                 lr,
                 warm_up,
                 batch_size,
                 embedding_dim,
                 recurrent_units,
                 iterations,
                 save_interval,
                 use_gru,
                 pretrained_model_path=''):
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.model_name = model_name
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.recurrent_units = recurrent_units
        self.iterations = iterations
        self.save_interval = save_interval
        self.use_gru = use_gru
        self.pretrained_model_path = pretrained_model_path


class ChatLSTM(CheckpointManager):
    def __init__(self, config, evaluate):
        super().__init__()
        assert config.save_interval >= 1000
        self.train_data_path = config.train_data_path
        self.validation_data_path = config.validation_data_path
        self.model_name = config.model_name
        self.lr = config.lr
        self.warm_up = config.warm_up
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.recurrent_units = config.recurrent_units
        self.iterations = config.iterations
        self.save_interval = config.save_interval
        self.use_gru = config.use_gru
        self.pretrained_model_path = config.pretrained_model_path
        self.pretrained_iteration_count = 0

        self.set_model_name(config.model_name)
        if evaluate:
            self.set_global_seed()

        self.model = None
        self.pretrained_model_output_size = 0
        self.pretrained_vocab_size = 0
        if self.pretrained_model_path != '':
            if self.is_model_path_exists(self.pretrained_model_path):
                self.model = tf.keras.models.load_model(self.pretrained_model_path, compile=False)
                self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.pretrained_model_path)
                self.pretrained_model_output_size, self.pretrained_vocab_size = self.get_output_size_and_vocab_size_from_model(self.model)
                print(f'load pretrained model success : {self.pretrained_model_path}')
            else:
                print(f'pretrained model not found : {self.pretrained_model_path}')
                exit(0)

        self.train_data_generator = DataGenerator(
            data_path=self.train_data_path,
            batch_size=self.batch_size,
            pretrained_model_output_size=self.pretrained_model_output_size,
            pretrained_vocab_size=self.pretrained_vocab_size,
            training=True)
        self.validation_data_generator = DataGenerator(
            data_path=self.validation_data_path,
            batch_size=self.batch_size,
            pretrained_model_output_size=self.pretrained_model_output_size,
            pretrained_vocab_size=self.pretrained_vocab_size)

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def is_model_path_exists(self, model_path):
        return os.path.exists(model_path) and os.path.isfile(model_path) and model_path.endswith('.h5')

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

    def predict(self, model, nl):
        encoder_x = self.train_data_generator.preprocess(nl, target='sequence_pad')
        encoder_x = np.asarray(encoder_x).reshape((1,) + encoder_x.shape)
        decoder_x = self.train_data_generator.tokenizer.get_bos_sequence()
        decoder_x = np.asarray(decoder_x).reshape((1,) + decoder_x.shape)
        output_nl = ''
        for i in range(self.train_data_generator.tokenizer.max_sequence_length - 1):
            y = self.graph_forward(model, [encoder_x, decoder_x])
            index, word, end = self.train_data_generator.postprocess(np.array(y[0]))
            if end:
                break
            if i == 0:
                output_nl = f'{word}'
            else:
                output_nl += f' {word}'
            decoder_x[0][i+1] = index
        return output_nl

    def compile(self, model):
        model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['acc'])

    def get_output_size_and_vocab_size_from_model(self, model):
        output_size = model.input_shape[-1][-1]
        vocab_size = model.output_shape[-1]
        return output_size, vocab_size

    def evaluate(self, dataset='train', data_path='', chat=False):
        if chat:
            self.train_data_generator.prepare(load_utterances=False)
            print('chat start\n')
            while True:
                nl = input('Me : ')
                output_nl = self.predict(self.model, nl)
                print(f'AI : {output_nl}')
        else:
            data_generator = None
            if data_path == '':
                assert dataset in ['train', 'validation']
                self.train_data_generator.prepare()
                if dataset == 'train':
                    data_generator = self.train_data_generator
                else:
                    self.validation_data_generator.prepare()
                    self.validation_data_generator.tokenizer = self.train_data_generator.tokenizer  # use trained tokenizer for validation
                    data_generator = self.validation_data_generator
            else:
                self.train_data_generator.prepare()
                data_generator = DataGenerator(
                    data_path=data_path,
                    batch_size=self.batch_sze,
                    pretrained_model_output_size=self.pretrained_model_output_size,
                    pretrained_vocab_size=self.pretrained_vocab_size)
                data_generator.prepare()
                data_generator.tokenizer = self.train_data_generator.tokenizer  # use trained tokenzer for validation:w
            self.compile(self.model)
            bos_ret = self.model.evaluate(
                x=data_generator.evaluate_generator(evaluate_bos=True),
                batch_size=self.batch_size,
                return_dict=True)
            bos_acc = bos_ret['acc']
            random_ret = self.model.evaluate(
                x=data_generator.evaluate_generator(evaluate_bos=False),
                batch_size=self.batch_size,
                return_dict=True)
            random_acc = random_ret['acc']
            acc_hm = (bos_acc * random_acc * 2.0) / (bos_acc + random_acc + 1e-7)
            print(f'bos_acc : {bos_acc:.4f}, random_acc : {random_acc:.4f}, acc_hm : {acc_hm:.4f}')
            return bos_acc, random_acc, acc_hm

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def train(self):
        self.train_data_generator.prepare()
        self.validation_data_generator.prepare()
        if self.model is None:
            data_max_sequence_length = self.train_data_generator.tokenizer.max_sequence_length
            data_vocab_size = self.train_data_generator.tokenizer.vocab_size
            self.model = Model(
                max_sequence_length=data_max_sequence_length,
                vocab_size=data_vocab_size,
                embedding_dim=self.embedding_dim,
                recurrent_units=self.recurrent_units,
                use_gru=self.use_gru).build()

        self.compile(self.model)
        self.model.summary()
        print()
        print(f'vocab size : {self.train_data_generator.tokenizer.vocab_size}')
        print(f'max sequence length : {self.train_data_generator.tokenizer.max_sequence_length}')
        print(f'train on {len(self.train_data_generator.json_paths)} samples.')
        print('start training')
        self.init_checkpoint_dir()
        iteration_count = self.pretrained_iteration_count
        compute_gradient = tf.function(self.compute_gradient)
        lr_scheduler = LRScheduler(lr=self.lr, lrf=0.1, iterations=self.iterations, warm_up=self.warm_up, policy='onecycle')
        eta_calculator = ETACalculator(iterations=self.iterations)
        eta_calculator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(self.optimizer, iteration_count)
            loss = compute_gradient(self.model, self.optimizer, self.loss_fn, batch_x, batch_y)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count)
            if iteration_count % self.save_interval == 0:
                print()
                bos_acc, random_acc, acc_hm = self.evaluate()
                content = f'_bos_acc_{bos_acc:.4f}_random_acc_{random_acc:.4f}'
                self.save_best_model(self.model, iteration_count, content=content, metric=acc_hm)
                print()
            if iteration_count == self.iterations:
                print('train end successfully')
                return


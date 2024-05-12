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
import argparse

from chatlstm import TrainingConfig, ChatLSTM


if __name__ == '__main__':
    config = TrainingConfig(
        train_data_path='/train_data/chatbot/train.csv',
        validation_data_path='/train_data/chatbot/validation.csv',
        model_name='model',
        lr=0.01,
        warm_up=0.1,
        batch_size=32,
        embedding_dim=32,
        recurrent_units=32,
        iterations=50000,
        save_interval=2500,
        use_gru=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--chat', action='store_true', help='chat in cli with pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate using given dataset')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='json or csv path for prediction or evaluation')
    args = parser.parse_args()
    if args.model != '':
        config.pretrained_model_path = args.model
    chatlstm = ChatLSTM(config=config, evaluate=args.evaluate)
    if args.evaluate:
        chatlstm.evaluate(dataset=args.dataset, data_path=args.path, chat=args.chat)
    else:
        chatlstm.train()


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/cfg.yaml', help='path of training configuration file')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--evaluate', action='store_true', help='evaluate using given dataset')
    parser.add_argument('--chat', action='store_true', help='chat in cli with pretrained model')
    parser.add_argument('--auto', action='store_true', help='auto chat flag using train data')
    parser.add_argument('--auto-count', type=int, default=10, help='auto chat count')
    parser.add_argument('--dataset', type=str, default='train', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='json data path for prediction or evaluation')
    args = parser.parse_args()
    cfg = TrainingConfig(cfg_path=args.cfg)
    if args.model != '':
        cfg.set_config('pretrained_model_path', args.model)
    if args.path != '':
        cfg.set_config('validation_data_path', args.path)
    chatlstm = ChatLSTM(cfg=cfg, evaluate=args.evaluate)
    if args.evaluate:
        chatlstm.evaluate(dataset='validation' if args.path != '' else args.dataset, chat=args.chat, chat_auto=args.auto, auto_count=args.auto_count)
    else:
        chatlstm.train()


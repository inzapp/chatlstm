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
import argparse

from tokenizer import Tokenizer
from ckpt_manager import CheckpointManager
from chatlstm import TrainingConfig, ChatLSTM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained checkpoint dir path')
    parser.add_argument('--auto', action='store_true', help='auto chat flag using train data')
    parser.add_argument('--auto-count', type=int, default=10, help='auto chat count')
    parser.add_argument('--dataset', type=str, default='train', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='json data path for evaluation or auto chatting')
    args = parser.parse_args()
    if not (os.path.exists(args.checkpoint) and os.path.isdir(args.checkpoint)):
        print(f'invalid checkpoint path : {args.checkpoint}')

    cfg_path = f'{args.checkpoint}/cfg.yaml'
    cfg = TrainingConfig(cfg_path=cfg_path)

    cm = CheckpointManager()
    model_path = cm.get_best_model_path(args.checkpoint)
    if model_path is None:
        model_path = cm.get_last_model_path(args.checkpoint)
        if model_path is None:
            print(f'no model found in {args.checkpoint}')
            exit(0)

    cfg.set_config('pretrained_model_path', model_path)
    cfg.set_config('pretrained_tokenizer_path', f'{args.checkpoint}/{Tokenizer.default_file_name}')
    if args.path != '':
        cfg.set_config('validation_data_path', args.path)
        args.dataset = 'validation'
    chatlstm = ChatLSTM(cfg=cfg, evaluate=True)
    chatlstm.chat(auto=args.auto, auto_count=args.auto_count, dataset=args.dataset)


# Copyright (c) 2020-2021 Joppe Geluykens
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from argparse import Namespace


def add_denoising_args(train_args: Namespace):
    train_args.replace_length = 1

    return train_args


def add_train_args(train_args: Namespace):
    # train_args.max_epoch = 10
    train_args.max_update = 10

    # Batch configuration
    #  effective batch size = num_gpus * max_sentences * update_freq
    train_args.update_freq = [16]  # increase for larger batch size
    # train_args.max_sentences = 10
    train_args.max_sentences_valid = 2  # 6 12
    train_args.required_batch_size_multiple = 1
    # Memory management:
    train_args.max_tokens = 2048
    train_args.tokens_per_sample = 512

    # Data
    train_args.sample_break_mode = "none"
    train_args.truncate_source = True
    train_args.truncate_sequence = True

    # Model config
    train_args.share_decoder_input_output_embed = True
    train_args.layernorm_embedding = True
    train_args.share_all_embeddings = False

    # Optimization config
    train_args.criterion = "label_smoothed_cross_entropy"
    train_args.optimizer = "adam"
    train_args.adam_betas = "(0.9, 0.98)"
    train_args.adam_eps = 1e-08
    train_args.lr = [1e-04]  # 3e-05
    train_args.lr_scheduler = "inverse_sqrt"
    train_args.warmup_updates = 1000
    train_args.warmup_init_lr = 1e-08
    train_args.weight_decay = 0.01
    train_args.clip_norm = 0.0
    train_args.label_smoothing = 0.1
    train_args.dropout = 0.1
    train_args.attention_dropout = 0.1

    train_args.seed = 97

    return train_args

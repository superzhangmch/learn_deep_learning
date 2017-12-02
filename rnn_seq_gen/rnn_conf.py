#encoding: utf8
import os
import sys
import time
import tensorflow as tf
import random
from build_sample import read_samples, get_samples
from rnn_model import RnnSeqGenModel

# == 模型参数
max_in_seq_size = 100 # RNN展开后的长度
max_line = 10000 * 2
batch_size = 64  # batch size
epoch_cnt = 100  # 一共多少个epoch
begin_lr = 0.01  #学习率
batch_size = 64  # batch size

save_path = "/home/work/zmc/song/rnn_rec/model" # 模型存储地址
save_prefix = "song" # 模型存储文件的文件名的前缀

#model = RnnSeqGenModel(in_dict_size    = 375053, #10000,
model = RnnSeqGenModel(in_dict_size    = 10000,
                       in_emb_size     = 64,
                       rnn_type        = "lstm",
                       rnn_layer_cnt   = 2,
                       rnn_hidden_size = 128,
                       num_sampled     = 1000,
                       learning_rate   = begin_lr)

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

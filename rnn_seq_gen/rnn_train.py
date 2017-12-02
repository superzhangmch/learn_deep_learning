#encoding: utf8
import os
import sys
import time
import tensorflow as tf
import random
from build_sample import read_samples, get_samples
from rnn_conf import model, begin_lr, save_path, save_prefix, batch_size

# == 模型参数
max_in_seq_size = 100 # RNN展开后的长度
max_line = 10000 * 50
epoch_cnt = 10   # 一共多少个epoch
save_every_batch_cnt = 3000

load_vali = False

# ======================================
print "prepare samples for train"
samples = read_samples("./sample_1w", max_line, max_in_seq_size)
train_samples, train_labels, train_seq_len = get_samples(samples, batch_size)
if load_vali:
    print "prepare samples for vali"
    samples = read_samples("./vali.data", max_line, max_in_seq_size)
    vali_samples,  vali_labels, vali_seq_len = get_samples(samples,  batch_size)
print "prepare samples done"

init_state_val = model.gen_init_state(batch_size)

cur_lr = begin_lr
for i in xrange(epoch_cnt):
    t_s = time.time()
    total_cnt = 0.
    total_loss = 0.

    total_step = len(train_samples)
    for j in xrange(len(train_samples)):
        if j > 0 and j % save_every_batch_cnt == 0:
            cur_lr = begin_lr * (0.825 ** (j / 3000))
            model.set_lr(cur_lr)

            model.save_model("%s/%s_%d_%d" % (save_path, save_prefix, i, j))
        # 训练
        feed_data = model.train_feed_dict(train_samples[j], train_labels[j], train_seq_len[j], init_state_val)
        req = [model.opt, model.out_loss]
        _, charrnn_loss = model.sess.run(req, feed_dict = feed_data) 
        total_cnt += 1
        total_loss += charrnn_loss
        print "%d/%d->%.4f, %.4f" % (j, total_step, charrnn_loss, total_loss / total_cnt)
        sys.stdout.flush()

        if j % 300 == 0:
            mean_loss = 1. * total_loss / total_cnt
            print "\nepoch=%d step=%d tm=%.2fs loss=%.5f lr=%f" % (i, j, time.time() - t_s, mean_loss, cur_lr)
            total_cnt = 0
            total_loss = 0.
            t_s = time.time()
            if not load_vali:
                continue
            cnt = len(vali_samples)
            start = random.randint(0, cnt / 2)
            total_cnt1 = 0
            total_loss1 = 0.
            for k in xrange(start, start + 10, 1):
                feed_data = model.train_feed_dict(vali_samples[k], vali_labels[k], vali_seq_len[k], init_state_val)
                req = [model.out_loss, model.out_final_rnn_state]
                charrnn_loss, pred_state = model.sess.run(req, feed_dict = feed_data) 
                total_cnt1 += 1
                total_loss1 += charrnn_loss
            print "valid %d/%d loss=%.4f" % (i, j, total_loss1 / total_cnt1)

    model.save_model("%s/%s_%d" % (save_path, save_prefix, i))

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

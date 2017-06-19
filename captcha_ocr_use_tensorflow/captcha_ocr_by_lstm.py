#encoding: utf8

import os
import sys
import tensorflow as tf
import numpy as np
import random
import datetime


from basic_fun import img_w, img_h, same_cnt
from make_sample import get_sample, get_test_data

# ===============================================

# M 代表对于验证码，按行方向为时间方向，按列为时间点做lstm的时候，每多少列作为一个时间点
# 一共80列，经试验，10或8列作为一个时间点，也就是一共10或8个时间点，效果都是很好的, 可以95%的效果

M = 8
def captcha_model():
    hid_size = 128

    # == rnn
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            hid_size, forget_bias=0.0, state_is_tuple=True)

    #def attn_cell():
    #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.6)
    attn_cell = lstm_cell
    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(2)], state_is_tuple=True)

    inputs = tf.placeholder("float", (None, img_w, img_h))
    inputs_seq_len = tf.placeholder(tf.int32, (None))

    inputs_1 = tf.reshape(inputs, (-1, img_w/M, img_h*M))
    # dynamic_rnn 的文档说sequence_length需要vector,其实 tf的vector就是 rank=1 的 tensor
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_1, dtype="float", sequence_length=inputs_seq_len)

    # == output
    out_size = 11 #  10个数字再加额外1个blank类别
    output = tf.reshape(outputs, [-1, hid_size])

    softmax_w = tf.Variable(tf.truncated_normal([hid_size, out_size], stddev=0.1,dtype=tf.float32),name='W')
    softmax_b = tf.Variable(tf.constant(0., dtype = tf.float32,shape=[out_size],name='b'))

    logits_out = tf.matmul(output, softmax_w) + softmax_b
    logits_out = tf.reshape(logits_out, (-1, img_w/M, out_size))
    logits_out = tf.transpose(logits_out, (1, 0, 2))
    return inputs, inputs_seq_len, logits_out

def captcha_train(logits_inputs, seq_len):
    labels = tf.sparse_placeholder("int32")
    loss = tf.nn.ctc_loss(labels=labels, inputs=logits_inputs, sequence_length=seq_len)
    loss = tf.reduce_mean(loss)
    return labels, loss

def captcha_predict(logits_inputs, seq_len):
    #ret = tf.nn.ctc_greedy_decoder(inputs=inputs, sequence_length=seq_len, merge_repeated=False)
    decoded = tf.nn.ctc_beam_search_decoder(inputs=logits_inputs, sequence_length=seq_len, merge_repeated=False)
    return decoded

model_inputs, inputs_seq_len, logits_output = captcha_model()
model_labels, loss = captcha_train(logits_output, inputs_seq_len)
decode_out = captcha_predict(logits_output, inputs_seq_len)

optimizer = tf.train.AdamOptimizer(learning_rate=.001) # 学习率实验在这个时候表现较好
opt = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 128
epoch_cnt = 10000
mean_cnt = batch_size
loss_sum = 0.
loss_cnt = 0
saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)

test_labels, test_samples, test_pred_labels = get_test_data()

for ii in xrange(epoch_cnt):
    for j in xrange(1000):
        samples = []
        labels = [[], [], (batch_size, 4)]
        pred_labels = []
        for i in xrange(batch_size):
            [a1, a2, a3, a4], img_data = get_sample()
            samples.append(img_data)
            labels[0] += [[i, 0], [i, 1], [i, 2], [i, 3]]
            labels[1] += [a1, a2, a3, a4]
            pred_labels.append(["%d%d%d%d" % (a1, a2, a3, a4), "", 0])

        if j > 0 and j % mean_cnt == 0:
            print ""
            ok_cnt = 0
            ret = sess.run(decode_out, feed_dict = {model_inputs: samples, inputs_seq_len: [img_w/M]*batch_size})
            aaa = ret[0][0].indices
            bbb = ret[0][0].values
            for k in xrange(len(aaa)):
                idx = aaa[k][0]
                pred_labels[idx][1] += str(bbb[k])
                pred_labels[idx][2] += 1
            for k in xrange(len(pred_labels)):
                if pred_labels[k][0] == pred_labels[k][1]:
                    ok_cnt += 1
            for k in xrange(10):
                kk = random.randint(0, batch_size - 1)
                eq = same_cnt(pred_labels[kk][0], pred_labels[kk][1])
                print "train:", pred_labels[kk][0], pred_labels[kk][1], eq

            now = datetime.datetime.now()
            print "[%d:%d:%d] r=%d, step=%d, loss=%.4f, train_ok_rate=%d/%d=%.2f%%" % (now.hour,now.minute,now.second, ii, j, loss_sum / loss_cnt, ok_cnt, batch_size, ok_cnt * 100. / batch_size)

            loss_sum = 0.
            loss_cnt = 0

            sys.stdout.flush()
        # print labels
        _, ls = sess.run([opt, loss], feed_dict = {model_inputs: samples, model_labels: labels, inputs_seq_len: [img_w/M]*batch_size})
        loss_sum += ls
        loss_cnt += 1
        print ".",
        sys.stdout.flush()

        if j > 0 and j % (batch_size * 5) == 0:
            print ""
            print "================ TEST ================"
            equal_cnt = 0
            samples = test_samples
            pred_labels = test_pred_labels
            test_cnt = len(test_samples)
            pred_labels_1 = [""] * test_cnt
            ret = sess.run(decode_out, feed_dict = {model_inputs: samples, inputs_seq_len: [img_w/M]*test_cnt})
            aaa = ret[0][0].indices
            bbb = ret[0][0].values
            for k in xrange(len(aaa)):
                kk = aaa[k][0]
                pred_labels_1[kk] += str(bbb[k])

            for k in xrange(test_cnt):
                if pred_labels[k] == pred_labels_1[k]:
                    equal_cnt += 1
            for k in xrange(10):
                kk = random.randint(0, test_cnt - 1)
                eq = same_cnt(pred_labels[kk], pred_labels_1[kk])
                print "Test:", pred_labels[kk], pred_labels_1[kk], eq
            print ii, j, "test_ok_rate = %.2f%% = %d/%d" % (100. * equal_cnt / test_cnt, equal_cnt, test_cnt)
            sys.stdout.flush()



#encoding:utf8
"""
四位数字验证码的识别，经试验，精度可达95%
"""
import os
import sys
import tensorflow as tf
import numpy as np
import random
import datetime


from basic_fun import img_w, img_h, same_cnt
from make_sample import get_sample, get_test_data

# ==============================

# 经实验，本例中 W 统一取1 或0.1有影响，影响在于那个W([76800, 512])的大维度的地方
def W(shape, default=.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=default))

# 经实验，本例中 B默认值取0或0.1基本无影响
def B(shape, default=.1):
    return tf.Variable(tf.constant(default, shape=shape))

def captcha_model(inputs):
    pool0 = inputs 
    # == cnn
    conv1 = tf.nn.conv2d(input=pool0, filter=W([5, 5, 1, 32]), strides=[1,1,1,1], padding="SAME")
    conv_res1 = tf.nn.relu(conv1 + B([32]))
    pool1 = tf.nn.max_pool(value=conv_res1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    conv2 = tf.nn.conv2d(input=pool1, filter=W([5, 5, 32, 32]), strides=[1,1,1,1], padding="SAME")
    conv_res2 = tf.nn.relu(conv2 + B([32]))
    pool2 = tf.nn.max_pool(value=conv_res2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    conv3 = tf.nn.conv2d(input=pool2, filter=W([3, 3, 32, 32]), strides=[1,1,1,1], padding="SAME")
    conv_res3 = tf.nn.relu(conv3 + B([32]))
    pool3 = tf.nn.max_pool(value=conv_res3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # == output
    pool3_shape = pool3.get_shape().as_list()[1:]
    dim = np.prod(pool3_shape) # == 76800
    input = tf.reshape(pool3, [-1, dim])
    # W([dim, 512], 0.005)):经试验，按relu初始化按stddev=sqrt(2/dim), 确实收敛更快
    input1 = tf.nn.relu(tf.matmul(input, W([dim, 512], 0.005)) + B([512]))

    output1 = tf.matmul(input1, W([512, 40], 0.05)) + B([40])
    logits = tf.reshape(output1, [-1, 4, 10])

    return logits

def captcha_train(labels, logits):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

def captcha_predict(logits):
    return tf.argmax(logits, axis=len(logits.shape) -1)

inputs = tf.placeholder("float", (None, img_w, img_h))
labels = tf.placeholder("int32", (None, 4))

inputs_0 = tf.reshape(inputs, [-1, img_w, img_h, 1])
logits_0 = captcha_model(inputs_0)
loss = captcha_train(labels, logits_0)
predict = captcha_predict(logits_0)

# 学习率用更大比如0.1,0.01等的时候，收敛上不行，记得好像是不收敛。总之对于学习率可以加减几个0的方式广泛尝试
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
opt = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# =====================================

batch_size = 128
epoch_cnt = 10000
mean_cnt = 99
loss_sum = 0.
loss_cnt = 0
test_labels, test_samples, test_pred_labels = get_test_data()

saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)

for ii in xrange(epoch_cnt):
    for j in xrange(1000):
        samples = []
        labels_list = []
        pred_labels = []
        pred_labels_1 = []
        for i in xrange(batch_size):
            [a1, a2, a3, a4], img_data = get_sample()
            samples.append(img_data)
            labels_list.append([a1, a2, a3, a4])
            pred_labels.append("%d%d%d%d" % (a1, a2, a3, a4))

        if j > 0 and j % mean_cnt == 0:
            print ""
            ret = sess.run(predict, feed_dict = {inputs: samples})
            for k in xrange(len(ret)):
                pred_labels_1.append("%d%d%d%d" % (ret[k][0], ret[k][1], ret[k][2], ret[k][3]))
            ok_cnt = 0
            for k in xrange(batch_size):
                if pred_labels[k] == pred_labels_1[k]:
                    ok_cnt += 1
            for k in xrange(5):
                eq = same_cnt(pred_labels[k], pred_labels_1[k])
                print "train:", pred_labels[k], pred_labels_1[k], eq
                eq = same_cnt(pred_labels[-k-1], pred_labels_1[-k-1])
                print "train:", pred_labels[-k-1], pred_labels_1[-k-1], eq
            now = datetime.datetime.now()
            print "[%d:%d:%d] r=%d, step=%d, loss=%.4f, ok=%d/%d=%.2f%%" % (now.hour,now.minute,now.second, ii, j, loss_sum / loss_cnt, ok_cnt, batch_size, ok_cnt * 100. / batch_size)

            loss_sum = 0.
            loss_cnt = 0

            sys.stdout.flush()

        # print labels
        _, ls = sess.run([opt, loss], feed_dict = {inputs: samples, labels: labels_list})
        loss_sum += ls
        loss_cnt += 1
        print ".",
        sys.stdout.flush()

        if j % 128 * 5 == 0:
            print ""
            print "================ TEST ================"
            equal_cnt = 0
            samples = test_samples
            pred_labels = test_pred_labels
            pred_labels_1 = []
            ret = sess.run(predict, feed_dict = {inputs: samples})
            for k in xrange(len(ret)):
                pred_labels_1.append("%d%d%d%d" % (ret[k][0], ret[k][1], ret[k][2], ret[k][3]))
            for k in xrange(len(ret)):
                if pred_labels[k] == pred_labels_1[k]:
                    equal_cnt += 1
            for k in range(10):
                eq = same_cnt(pred_labels[k], pred_labels_1[k])
                print "Test:", pred_labels[k], pred_labels_1[k], eq
            print ii, j, "ok_rate = %.2f%% = %d/%d" % (1. * equal_cnt / 512 * 100, equal_cnt, 512)
            sys.stdout.flush()
    #saver.save(sess, "/home/work/zmc/aa/model/cnn_model_%d" % (ii))

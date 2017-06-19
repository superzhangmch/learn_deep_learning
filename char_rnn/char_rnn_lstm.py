#encoding: utf8
import os
import sys
import time
import tensorflow as tf
import numpy as np
import random

# == 模型参数
in_seq_size = 64 # RNN展开后的长度
emb_size = 32 # 模型词向量维度
in_cls_cnt = (0x7E - 0x20 + 1 + 0x0d - 0x09 + 1) + 1 # 模型输入字符数. 要求输入数据都是可打印字符外加\t,\n等
layer_cnt = 2 # 模型多层RNN层数
rnn_hid_size = 128 # 模型 lstm 隐层大小
out_size = in_cls_cnt # 模型输出字符数

# == 训练还是预测
is_train = False

# == 训练参数
begin_lr = 0.01 # 学习率
batch_size = 64 # batch size
epoch_step_cnt = 10000 # 一个epoch 共包含多少次训练
epoch_cnt = 100  # 一共多少个epoch
train_data_path = "linux_src/" # 样本数据的存放路径. 其下面需要都是文本文件
save_path = "/home/work/zmc/aa/model" # 模型存储地址
save_prefix = "cnn_model" # 模型存储文件的文件名的前缀

# == 预测用参数
prefix_for_txt_gen = "int main(int char" # 文本生成的时候，以此为前缀生成
predict_gen_len = 1000 # 一共生成多长的文本
load_epoch_id = 70 # 加载save下来的哪一个模型数据
temperature = 0.6 # 用于控制生成的多样性。lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

# ======================================

def charrnn_model():
    """ model """
    inputs = tf.placeholder(tf.int32, (None, None)) # shape == (batch_size, in_seq_size)
    inputs_seq_len = tf.placeholder(tf.int32, (None))

    # embeding 的实现方式，可以是tf.contrib.rnn.EmbeddingWrapper 可以是自己emb lookup，也可以是自己用全连接层实现
    #emb_w = tf.Variable(tf.truncated_normal([in_cls_cnt, emb_size], stddev=0.1, dtype=tf.float32), name='emb_W')
    #input_one_hot = tf.one_hot(inputs, in_cls_cnt)
    #input_reshape = tf.reshape(input_one_hot, (-1, in_cls_cnt))
    #inputs_1 = tf.matmul(input_reshape, emb_w)
    #inputs_1 = tf.reshape(inputs_1, (-1, tf.shape(inputs)[1], emb_size))

    # == rnn
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            rnn_hid_size, forget_bias=1.0, state_is_tuple=True)

    #def attn_cell():
    #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.6)
    attn_cell = lstm_cell
    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(layer_cnt)])

    inputs_1 = tf.reshape(inputs, (-1, tf.shape(inputs)[1], 1))
    cell = tf.contrib.rnn.EmbeddingWrapper(cell, embedding_classes=in_cls_cnt, embedding_size=emb_size, 
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

    batch_sz = tf.placeholder(tf.int32, (None))
    init_state = cell.zero_state(batch_sz, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_1, dtype="float", 
                              sequence_length=inputs_seq_len, initial_state=init_state)

    # == output
    output = tf.reshape(outputs, [-1, rnn_hid_size])

    softmax_w = tf.Variable(tf.truncated_normal([rnn_hid_size, out_size], stddev=0.1,dtype=tf.float32),name='W')
    softmax_b = tf.Variable(tf.constant(0., dtype = tf.float32,shape=[out_size],name='b'))

    def do_predict():
        max_step_cnt = tf.placeholder(tf.int32, ())
        idx = tf.constant(0)
        z = tf.Variable(0, tf.int32)
        cell_input = tf.zeros((batch_sz), dtype=tf.int64)
        out_ids = tf.zeros((z, batch_sz), dtype=tf.int64)
        out_probs = tf.zeros((z, batch_sz, out_size))

        def cond(idx, *_):
            return idx < max_step_cnt

        def body(idx, cell_input, cell_state, out_ids, out_probs):
            with tf.variable_scope("rnn"):
                tf.get_variable_scope().reuse_variables()
                cur_out, cur_state = cell(cell_input, cell_state)
            # cur_out.shape == (batch_size, rnn_hid_size)

            logits = tf.matmul(cur_out, softmax_w) + softmax_b
            cur_prob = tf.nn.softmax(logits / temperature)
            # cur_prob.shape == (batch_size, out_size)

            out = tf.multinomial(tf.log(cur_prob), 1)
            cur_id = tf.reshape(out, (-1,))

            #cur_id = tf.argmax(cur_prob, axis=-1)
            # cur_id.shape == (batch_size)

            cur_out_ids = tf.concat([out_ids, tf.expand_dims(cur_id, 0)], axis=0)
            cur_probs = tf.concat([out_probs, tf.expand_dims(cur_prob, 0)], axis=0)
            return idx+1, cur_id, cur_state, cur_out_ids, cur_probs
        wl = tf.while_loop(cond, body, loop_vars=[idx, cell_input, init_state, out_ids, out_probs])
        return max_step_cnt, cell_input, wl
    predict = do_predict()

    logits_out = tf.matmul(output, softmax_w) + softmax_b
    logits_out = tf.reshape(logits_out, (-1, tf.shape(inputs)[1], out_size))
    return inputs, inputs_seq_len, batch_sz, init_state, state, predict, logits_out

def charrnn_train(logits_inputs):
    labels = tf.placeholder(tf.int32, (None))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_inputs)
    loss = tf.reduce_mean(loss)
    return labels, loss

def charrnn_predict(logits_inputs):
    decoded = tf.nn.softmax(logits=logits_inputs)
    return decoded

model_inputs, inputs_seq_len, batch_sz, init_state, final_state, predict, logits_output = charrnn_model()
model_labels, loss = charrnn_train(logits_output)
decode_out = charrnn_predict(logits_output)
max_step_cnt, cell_input, while_loop = predict

lr_rate = tf.Variable(begin_lr, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
opt = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

id2char = [chr(i) for i in xrange(0x20, 0x7E +1 , 1)] + [chr(i) for i in xrange(0x09, 0x0d +1 , 1)] + [chr(0)]
char2id = {id2char[i]: i for i in xrange(len(id2char))}

# prepare samples 
def get_samples(batch_size, in_seq_size, step_cnt):
    path = train_data_path
    max_size = batch_size * in_seq_size * step_cnt
    files = None
    for root, sub_folder, file_list in os.walk(path):
        files = file_list
        break
    random.shuffle(files)
    content = ""

    for i in xrange(len(files)):
        buf = open(path + files[i], "rb").read()
        buf = "".join([char for char in buf if char in char2id])
        content += buf + "\0"
        if len(content) >= max_size + 1:
            break

    content = [char2id[char] for char in content]

    samples = content[0: max_size]
    labels = content[1: max_size + 1]
    re_shape = (batch_size, step_cnt, in_seq_size)
    return np.reshape(samples, re_shape).transpose((1,0,2)), np.reshape(labels, re_shape).transpose((1,0,2))


init_state_val_1 = sess.run(init_state, feed_dict={batch_sz: 1})

saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)

if not is_train:
    epoch = load_epoch_id
    saver.restore(sess, "%s/%s_%d" % (save_path, save_prefix, epoch))
    max_cnt = predict_gen_len
    gen_prefix = prefix_for_txt_gen
    # ==============================
    pred_state = init_state_val_1
    feed_data = {batch_sz: 1, model_inputs: [[char2id[c] for c in gen_prefix[:-1]]],
                 inputs_seq_len: [len(gen_prefix) - 1], }
    for k in xrange(len(init_state)):
        feed_data[init_state[k][0]] = pred_state[k][0]
        feed_data[init_state[k][1]] = pred_state[k][1]
    decode_res, f_state = sess.run([decode_out, final_state], feed_dict = feed_data) 
    # ==============================
    last_id = char2id[gen_prefix[-1]]
    pred_feed_data = {batch_sz:1, max_step_cnt: max_cnt, cell_input:[last_id]}
    for k in xrange(len(init_state)):
        pred_feed_data[init_state[k][0]] = f_state[k][0]
        pred_feed_data[init_state[k][1]] = f_state[k][1]
    wl_res = sess.run(while_loop, feed_dict = pred_feed_data)
    decode_res = np.array(wl_res[3]).reshape((-1))
    decode_chars = [id2char[id] for id in decode_res]
    gen_str = "".join(decode_chars)
    print "================================="
    print gen_prefix + gen_str
    sys.exit(0)

samples, labels = get_samples(batch_size, in_seq_size, epoch_step_cnt)
init_state_val = sess.run(init_state, feed_dict={batch_sz: batch_size})
state = init_state_val

for i in xrange(epoch_cnt):
    t_s = time.time()
    total_cnt = 0.
    total_loss = 0.

    # 即使用了 AdamOptimizer, 仍然做lr衰减(不过测试后没发现特别效果)
    cur_lr = begin_lr * (0.9 ** i)
    sess.run(tf.assign(lr_rate, cur_lr))

    for j in xrange(epoch_step_cnt):
        # 训练
        feed_data = {batch_sz: batch_size, model_inputs: samples[j],
                                           model_labels: labels[j],
                                           inputs_seq_len: [in_seq_size]*batch_size, }
        for k in xrange(len(init_state)):
            feed_data[init_state[k][0]] = state[k][0]
            feed_data[init_state[k][1]] = state[k][1]
        _, charrnn_loss, state = sess.run([opt, loss, final_state], feed_dict = feed_data) 
        total_cnt += 1
        total_loss += charrnn_loss
        print ".",
        sys.stdout.flush()

        if j > 0 and j % 100 == 0:
            mean_loss = 1. * total_loss / total_cnt
            print "\nepoch=%d step=%d tm=%.2fs loss=%.5f lr=%f" % (i, j, time.time() - t_s, mean_loss, cur_lr)
            total_cnt = 0
            total_loss = 0.
            # 对于当前训练数据，查看预测情况
            decode_res, f_state = sess.run([decode_out, final_state], feed_dict = feed_data) 
            decode_res = sess.run(tf.argmax(decode_res[0], -1))
            aa = "".join([id2char[ii] if id2char[ii] != '\n' else "\\n" for ii in samples[j][0]])
            bb = "".join([id2char[ii] if id2char[ii] != '\n' else "\\n" for ii in decode_res])
            print "<<<-----"
            print "input=|%s|" % (aa)
            print "-----"
            print "output=|%s|" % (bb)
            print "----->>>"
            t_s = time.time()

            # ==== predict 
            max_cnt = 500
            last_id = char2id[bb[-1]]
            pred_state = init_state_val_1
            tm_start = time.time()
            # ==============================
            # 作测试
            pred_feed_data = {batch_sz:1, max_step_cnt: max_cnt, cell_input:[last_id]}
            for k in xrange(len(init_state)):
                pred_feed_data[init_state[k][0]] = pred_state[k][0]
                pred_feed_data[init_state[k][1]] = pred_state[k][1]
            wl_res = sess.run(while_loop, feed_dict = pred_feed_data)
            decode_res = np.array(wl_res[3]).reshape((-1))
            decode_chars = [id2char[id] for id in decode_res]
            gen_str = "".join(decode_chars)
            # ==============================
            ## 用多次递归调用 time_step_size=1 的 rnn 的方式来作预测, 较慢
            #last_id = char2id[bb[-1]]
            #gen_str_1 = ""
            #for kk in xrange(max_cnt):
            #    pred_feed_data = {batch_sz: 1, model_inputs: [[last_id]], inputs_seq_len: [1]}
            #    for k in xrange(len(init_state)):
            #        pred_feed_data[init_state[k][0]] = pred_state[k][0]
            #        pred_feed_data[init_state[k][1]] = pred_state[k][1]
            #    decode_res, pred_state = sess.run([decode_out, final_state], feed_dict = pred_feed_data) 
            #    cur_id = sess.run(tf.argmax(tf.reshape(decode_res, [-1]), -1))
            #    #print "xxx_1", cur_id, id2char[cur_id]
            #    gen_str_1 += id2char[cur_id]
            #    last_id = cur_id
            # ==============================
            tm_used = time.time() - tm_start
            print i, j, "tm=%.2f gen: [%s]" %(tm_used, gen_str)
    saver.save(sess, "%s/%s_%d" % (save_path, save_prefix, i))

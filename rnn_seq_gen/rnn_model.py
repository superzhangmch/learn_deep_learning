#encoding: utf8
import os
import sys
import tensorflow as tf
import math

class RnnSeqGenModel(object):
    """
    RNN 的序列生成模型
    -- 一些成员变量说明
        self.fd_XXX: 都是要 feed 给 feed_dict 的
        self.out_XXX: 都是可以 sess.run(self.out_XXX, ...) 输出的
            self.fd_batch_size:
            self.fd_rnn_init_state:

            # for train
            self.fd_inputs:
            self.fd_inputs_seq_len:
            self.fd_labels:

            # for sequence generate:
            self.fd_seqgen_max_length:
            self.fd_seqgen_cell_inputs:
            self.fd_seqgen_temperature:

            self.out_final_rnn_state:
            self.out_decoded:
            self.out_loss:
            self.out_predict_loop
            self.sess
            self.opt: 
    """

    def __init__(self, in_dict_size, in_emb_size, \
                        rnn_type, rnn_layer_cnt, rnn_hidden_size, \
                        num_sampled, \
                        learning_rate):
        # == 模型参数
        self._in_dict_size = in_dict_size + 1 # 模型输入类别数
        self._in_emb_size = in_emb_size # 模型词向量维度

        self._num_sampled_val = num_sampled
        self._rnn_layer_cnt = rnn_layer_cnt # 模型多层RNN层数
        self._rnn_hid_size = rnn_hidden_size # 模型 lstm 隐层大小
        self._out_cls_size = self._in_dict_size # 模型输出字符数

        # == 训练参数
        self._begin_lr = learning_rate # 学习率

        self.func_gen_code4_rnn_cell()
        self.func_gen_code4_softmax_parameter()
        self.func_gen_code4_train()
        self.func_gen_code4_seq_generate()
        self.func_model_init()

    def func_gen_code4_rnn_cell(self):
        """ rnn cell """
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self._rnn_hid_size, forget_bias=1.0, state_is_tuple=True)

        #def attn_cell():
        #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.6)
        attn_cell = lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self._rnn_layer_cnt)])

        w_init = tf.truncated_normal_initializer(stddev=1. / math.sqrt(self._in_dict_size))
        self._rnn_cell = tf.contrib.rnn.EmbeddingWrapper(cell, embedding_classes=self._in_dict_size,
                                embedding_size=self._in_emb_size,
                                initializer=w_init)

        self.fd_batch_size = tf.placeholder(tf.int32) # batch size
        self.fd_rnn_init_state = self._rnn_cell.zero_state(self.fd_batch_size, tf.float32)

    def func_gen_code4_softmax_parameter(self):
        """ softmax parameter """
        sm_w_init = tf.truncated_normal([self._out_cls_size, self._rnn_hid_size],
                                             stddev=0.1, dtype=tf.float32)
        sm_b_init = tf.constant(0., dtype = tf.float32, shape=[self._out_cls_size])
        self._softmax_w = tf.Variable(sm_w_init, name='softmax_W')
        self._softmax_b = tf.Variable(sm_b_init, name='softmax_B')
    
    def func_gen_code4_train(self):
        self.fd_inputs = tf.placeholder(tf.int32, (None, None)) # shape == (batch_size, in_seq_len)
        self.fd_inputs_seq_len = tf.placeholder(tf.int32, (None,)) # shape == (batch_size,)
        self.fd_labels = tf.placeholder(tf.int64, (None, None)) # shape == (batch_size, in_seq_len)

        in_seq_len = tf.shape(self.fd_inputs)[1] # in_seq_len
        emb_inputs = tf.reshape(self.fd_inputs, (-1, in_seq_len, 1))

        # == rnn outputs
        rnn_outputs, final_rnn_state = tf.nn.dynamic_rnn(cell=self._rnn_cell,
                                           inputs=emb_inputs, dtype="float", 
                                           sequence_length=self.fd_inputs_seq_len, 
                                           initial_state=self.fd_rnn_init_state)
        # rnn_outputs.shape == (batch_size, in_seq_len, rnn_hid_size)
        #        reshape to == (batch_size * in_seq_len, rnn_hid_size)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self._rnn_hid_size])
        self.out_final_rnn_state = final_rnn_state

        # == begin to softmax

        logits_out = tf.matmul(rnn_outputs, tf.transpose(self._softmax_w)) + self._softmax_b
        logits_out = tf.reshape(logits_out, (-1, in_seq_len, self._out_cls_size))
        self.out_decoded = tf.nn.softmax(logits=logits_out)

        if self._num_sampled_val <= 0: # ** no sampeled softmax
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.fd_labels, logits=logits_out)
        else: # ** sampeled softmax
            labels = tf.reshape(self.fd_labels, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(weights = self._softmax_w,
                                              biases = self._softmax_b,
                                              labels = labels,
                                              inputs = rnn_outputs,
                                              num_sampled=self._num_sampled_val,
                                              num_classes=self._in_dict_size)

        mask = tf.sequence_mask(self.fd_inputs_seq_len)
        mask = tf.cast(mask, tf.float32)
        loss = tf.reshape(loss, tf.shape(mask))
        loss = loss * mask
        loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(self.fd_inputs_seq_len), tf.float32)
        self.out_loss = loss

    def func_gen_code4_seq_generate(self):
        """
        基于 tf.while_loop 的序列生成。不用它发现生成很慢
        """
        self.fd_seqgen_max_length = tf.placeholder(tf.int32)
        self.fd_seqgen_cell_inputs = tf.zeros((self.fd_batch_size), dtype=tf.int64)
        self.fd_seqgen_temperature = tf.placeholder(tf.float32)

        idx = tf.constant(0)
        z = tf.Variable(0, tf.int32)
        out_ids = tf.zeros((z, self.fd_batch_size), dtype=tf.int64)
        out_probs = tf.zeros((z, self.fd_batch_size, self._out_cls_size))

        def cond(idx, *_):
            return idx < self.fd_seqgen_max_length
    
        def body(idx, cell_inputs, cell_state, out_ids, out_probs):
            with tf.variable_scope("rnn"):
                tf.get_variable_scope().reuse_variables()
                cur_out, cur_state = self._rnn_cell(cell_inputs, cell_state)
            # cur_out.shape == (batch_size, self._rnn_hid_size)
    
            softmax_w_1 = tf.transpose(self._softmax_w)
            logits = tf.matmul(cur_out, softmax_w_1) + self._softmax_b
            # temperature: 用于控制生成的多样性。lower temperature will cause the model 
            #   to make more likely, but also more boring and conservative predictions. 
            #   Higher temperatures cause the model to take more chances and increase 
            #   diversity of results, but at a cost of more mistakes.
            cur_prob = tf.nn.softmax(logits / self.fd_seqgen_temperature)
            # cur_prob.shape == (batch_size, self._out_cls_size)
    
            out = tf.multinomial(tf.log(cur_prob), 1)
            cur_id = tf.reshape(out, (-1,))
    
            #cur_id = tf.argmax(cur_prob, axis=-1)
            # cur_id.shape == (batch_size)
    
            cur_out_ids = tf.concat([out_ids, tf.expand_dims(cur_id, 0)], axis=0)
            cur_probs = tf.concat([out_probs, tf.expand_dims(cur_prob, 0)], axis=0)
            return idx+1, cur_id, cur_state, cur_out_ids, cur_probs

        loop_vars=[idx, self.fd_seqgen_cell_inputs, self.fd_rnn_init_state, out_ids, out_probs]
        self.out_predict_loop = tf.while_loop(cond, body, loop_vars=loop_vars)

    def func_model_init(self):
        self._fd_lr_rate = tf.Variable(self._begin_lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._fd_lr_rate)
        opt = optimizer.minimize(self.out_loss)
        self.opt = opt
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    
    # =============================

    def gen_init_state(self, batch_size):
        """ 生成RNN的zero的state值, 以作为RNN的状态输入 """
        return self.sess.run(self.fd_rnn_init_state, feed_dict={self.fd_batch_size: batch_size})

    def set_lr(self, cur_lr):
        """ set learning rate """
        self.sess.run(tf.assign(self._fd_lr_rate, cur_lr))

    def save_model(self, path):
        """ save model """
        self._model_saver.save(self.sess, path)

    def restore_model(self, path):
        """ restore model """
        self._model_saver.restore(self.sess, path)

    def train_feed_dict(self, sample_data, labels, seq_len, state_init):
        """
        得到训练时需要的feed_dict 
        """
        batch_size = len(sample_data)
        feed_data = {self.fd_batch_size: batch_size,
                     self.fd_inputs: sample_data,
                     self.fd_labels: labels,
                     self.fd_inputs_seq_len: seq_len}
        for k in xrange(len(self.fd_rnn_init_state)):
            feed_data[self.fd_rnn_init_state[k][0]] = state_init[k][0]
            feed_data[self.fd_rnn_init_state[k][1]] = state_init[k][1]
        return feed_data

    def do_predict(self, input_seq, init_state_val):
        """
        进行序列预测. 输出预测的每个softmax概率值，以及RNN的最终状态输出
        input_seq: 输入序列
        init_state_val: RNN的状态输入
        """
        feed_data = self.train_feed_dict([input_seq], [[0]*len(input_seq)],
                                         [len(input_seq)], init_state_val)
        return self.sess.run([self.out_decoded, self.out_final_rnn_state], feed_dict = feed_data)

    def do_seqgen(self, max_cnt, temp_val, last_id, last_state):
        """
        进行序列生成。
        max_cnt: 最长生成这么长的序列
        temp_val: temperature 值，取值范围 0~1, 调节多样性。越小越缺乏多样性
        last_id: 从这个id开始进行生成
        last_state: 以它最为rnn的状态输入
        """
        pred_feed_data = {  self.fd_batch_size: 1,
                            self.fd_seqgen_max_length: max_cnt,
                            self.fd_seqgen_cell_inputs: [last_id],
                            self.fd_seqgen_temperature: temp_val}
        init_state = self.fd_rnn_init_state
        for k in xrange(len(init_state)):
            pred_feed_data[init_state[k][0]] = last_state[k][0]
            pred_feed_data[init_state[k][1]] = last_state[k][1]
        return self.sess.run(self.out_predict_loop, feed_dict = pred_feed_data)

if __name__ == "__main__":
    model = RnnSeqGenModel(in_dict_size = 10000,  
                            in_emb_size = 64, 
                            rnn_type = "lstm", 
                            rnn_layer_cnt = 2, 
                            rnn_hidden_size = 128, 
                            num_sampled = 1000,
                            learning_rate = 0.01)

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

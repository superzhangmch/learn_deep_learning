#encoding:utf8

import os
import pickle
import random

from basic_fun import img_w, img_h, load_data

# 下面目录存有测试集与样本集
train_data_path = "train_data"
train_test_path = "train_test"

# 为了加快load，把数据集dump为binary数据，使用的时候load
train_data_bin = "train.bin"
train_test_bin = "test.bin"

# ===================
# 加载样本
# 本例子使用了128 * 1000 个样本
# ===================
for_debug = False
if for_debug:
    train_data = load_data(train_data_path, 512)
    train_test = load_data(train_test_path, 121)
else:
    if not os.path.exists(train_data_bin):
        train_data = load_data(train_data_path)
        pickle.dump(train_data, open(train_data_bin, "w"))
    if not os.path.exists(train_test_bin):
        train_test = load_data(train_test_path)
        pickle.dump(train_test, open(train_test_bin, "w"))
    train_data = pickle.load(open(train_data_bin, "rb"))
    train_test = pickle.load(open(train_test_bin, "rb"))


# ===================
# 获得单个样本
# 本例子用了512个, 当然应该用更多更好
# ===================
train_data_list = [i for i in xrange(len(train_data[0]))]
sample_idx = -1
def get_sample():
    """ 获得单个样本 """
    global sample_idx
    sample_idx += 1
    sample_idx = sample_idx % len(train_data[0])
    if sample_idx == 0:
        random.shuffle(train_data_list)
    idx = train_data_list[sample_idx]
    return train_data[0][idx], train_data[1][idx]


# ===================
# 获得测试集合
# ===================
def get_test_data():
    """ 获得所有测试集 """
    test_pred_labels = []
    for i in xrange(len(train_test[0])):
        test_pred_labels.append(("%d" * len(train_test[0][i])) % tuple(train_test[0][i]))

    test_labels = train_test[0]
    test_samples = train_test[1]
    return test_labels, test_samples, test_pred_labels

if __name__ == "__main__":
    label, sample_data= get_sample()
    print label
    print get_test_data()[0]

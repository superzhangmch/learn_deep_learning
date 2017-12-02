#encoding: utf8
import os
import sys

# prepare samples 
def read_samples(file, max_line=0, max_in_seq_size=100):
    """ 
    file: 样本文件。每行一个序列样本，行内空格隔开，行格式"111 223 332\n"
    """
    arr = []
    ii = 0
    for line in open(file):
        LL = line.strip().split(" ")
        if not LL or len(LL) < 2:
            continue
        LL = [int(L) for L in LL if L]
        if len(LL) > max_in_seq_size:
            cnt = (len(LL) + max_in_seq_size - 1) / max_in_seq_size
            seq_size = len(LL) / cnt
            for i in xrange(cnt):
                LLL = LL[i * seq_size: (i+1) * seq_size + 1]
                if len(LLL) > 2:
                    arr.append(LLL)
        else:
            arr.append(LL)
        ii += 1
        if ii % 10000 == 0:
            print "read file=%s line=%d" % (file, ii)
        if max_line and ii > max_line:
            break
    return arr

def get_samples(samples, batch_size):
    batch_cnt = len(samples) / batch_size

    ret_samples = []
    ret_labels  = []
    ret_seq_len = []
    for i in xrange(batch_cnt):
        offset_from = i * batch_size
        offset_to   = offset_from + batch_size
        batch_samples = samples[offset_from: offset_to]
        arr_seq_len = [len(b) - 1 for b in batch_samples]
        max_len = max([len(b) - 1 for b in batch_samples])
        arr_samples = []
        arr_labels  = []
        for j in xrange(len(batch_samples)):
            arr = batch_samples[j]
            arr1 = arr[0:-1] + ([0]*(max_len - (len(arr) -1)))
            arr2 = arr[1:] + ([0]*(max_len - (len(arr) -1)))
            #print max_len, len(arr), len(arr1), len(arr2)
            arr_samples.append(arr1)
            arr_labels.append(arr2)
        ret_samples.append(arr_samples)
        ret_labels.append(arr_labels)
        ret_seq_len.append(arr_seq_len)
    return ret_samples, ret_labels, ret_seq_len


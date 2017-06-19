#encoding:utf8
import os
from PIL import Image
import numpy as np
import StringIO
import random

img_w = 80
img_h = 30
# ====================================


def get_img(file):
    """ 对图片做预处理 """
    img_obj = Image.open(file)
    # 彩色转灰度
    aa = img_obj.convert("L")
    aa = aa.resize((img_w, img_h), Image.BILINEAR)
    trans_img = aa
    shape = trans_img.size

    #trans_img.save("trans_img0.png")

    aa = trans_img.getdata()
    aa = list(aa)
    arr = np.array(aa)
    arr = arr.reshape([img_h, img_w])
    mm = arr.max()
    if mm <= 0:
        mm = 0
    # 除255也可以
    arr = (mm - arr) * 1. / mm
    arr = np.transpose(arr)

    # 往终端打印验证数据正误
    show = False
    if show:
        w, h = arr.shape
        for i in xrange(w):
            s = ""
            for j in xrange(h):
                if arr[i][j] < 0.05:
                    s += " "
                else:
                    s += "1"
            print s
        print arr.shape
    return arr


def gen_rand_captcha():
    """ 生成随机样本 """
    file_name = ""
    for i in xrange(4):
        idx = random.randint(0, 9)
        file_name += str(idx)
    buf = image.generate(file_name).read()
    return file_name, buf


def gen_rand_sample():
    """ """
    file_name, buf = gen_rand_captcha()
    aa = get_img(StringIO.StringIO(buf))
    return [int(num) for num in file_name], aa


def save_rand_sample(path, i):
    """ 保存随机样本为文件 """
    file_name, buf = gen_rand_captcha()
    open("%s/%d_%s.png" % (path, i, file_name), "w").write(buf)


def load_data(data_dir, max=-1):
    """ 加载最多max个训练样本或测试样本 """
    labels = []
    samples = []
    i = 0
    total = 0
    to_break = False
    for root, sub_folder, file_list in os.walk(data_dir):
        for file_path in file_list:
            i += 1
            if i % 1000 == 0:
                print i
            image_name = os.path.join(root,file_path)
            label = image_name.split("/")[-1].split(".")[0].split("_")[1]
            label = [int(char) for char in label]
            sample = get_img(image_name)
            samples.append(sample)
            labels.append(label)
            total += 1
            if max != -1 and total >= max:
                to_break = True
                break
        if to_break:
            break
    return labels, samples

def same_cnt(s1, s2):
    """ s1，s2的相同字符个数 """
    eq = 0
    for i in xrange(len(s1)):
        if len(s2) >= i + 1:
            if s1[i] == s2[i]:
                eq += 1
        else:
            break

    if eq == 0:
        return ""
    elif eq == len(s1):
        return "="
    else:
        return str(eq)


if __name__ == "__main__":
    # 生成样本
    from captcha.image import ImageCaptcha
    image = ImageCaptcha()

    for i in xrange(10):
        if i % 1000 == 0:
            print i
        save_rand_sample("test", i)

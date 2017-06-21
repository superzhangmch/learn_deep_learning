#encoding: utf8
import tensorflow as tf
import numpy as np
import scipy.io
from PIL import Image

# -- for train and test
learn_rate = 2. # 需要一个比较大学习率
step_cnt = 10000
save_every_cnt = 50
save_path = "save_path"

content_file = "building.png"
style_file = "fangao_sky.png"
noise_img_use_content_img = False # 数据初始化，是直接用随机初始，还是content image 噪声化得到

# -- for model
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
vgg_mat_file = "imagenet-vgg-verydeep-19.mat" # 下载地址 http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
content_fm_names = ["conv4_2"] # 原始论文就用的这层
style_fm_names = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"] # 原始论文就用的这些层
content_loss_weight = 0.001
style_loss_weight =  1.

class Vgg19Model(object):
    """
    from http://blog.csdn.net/matrix_space/article/details/54290460
    """
    # 该模型(或者说对应.mat模型文件)处理输入的方式就是原始rgb值减去平均值
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def build_net(self, ntype, nin, nwb=None):
        if ntype == 'conv':
            return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME') + nwb[1])
        elif ntype == 'pool':
            return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    def get_weight_bias(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

    def generate_noise_image(self, content_image, noise_ratio=0.6):
        """
        Returns a noise image intermixed with the content image at a certain ratio.
        """
        noise_image = np.random.uniform(-20, 20,
                (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype('float32')
        # White noise image from the content representation. Take a weighted average
        # of the values
        img = noise_image * noise_ratio + content_image * (1 - noise_ratio)
        return img

    def __init__(self, path):
        net = {}
        vgg_rawnet = scipy.io.loadmat(path) # should be a .mat file
        vgg_layers = vgg_rawnet['layers'][0]
    
        # 这里存放有,随机初始化的图片，训练过程就是优化它
        net['input'] = tf.Variable(tf.truncated_normal([1, IMAGE_HEIGHT, 
                       IMAGE_WIDTH, 3], mean=100., stddev=20., dtype=tf.float32)) # mean=100，因为图片数据处理就是
                                                                                  # rgb数值减去取值约为1百多的 MEAN 得到

        net['conv1_1'] = self.build_net('conv', net['input'], self.get_weight_bias(vgg_layers, 0))
        net['conv1_2'] = self.build_net('conv', net['conv1_1'], self.get_weight_bias(vgg_layers, 2))
        net['pool1'] = self.build_net('pool', net['conv1_2'])

        net['conv2_1'] = self.build_net('conv', net['pool1'], self.get_weight_bias(vgg_layers, 5))
        net['conv2_2'] = self.build_net('conv', net['conv2_1'], self.get_weight_bias(vgg_layers, 7))
        net['pool2'] = self.build_net('pool', net['conv2_2'])
    
        net['conv3_1'] = self.build_net('conv', net['pool2'], self.get_weight_bias(vgg_layers, 10))
        net['conv3_2'] = self.build_net('conv', net['conv3_1'], self.get_weight_bias(vgg_layers, 12))
        net['conv3_3'] = self.build_net('conv', net['conv3_2'], self.get_weight_bias(vgg_layers, 14))
        net['conv3_4'] = self.build_net('conv', net['conv3_3'], self.get_weight_bias(vgg_layers, 16))
        net['pool3'] = self.build_net('pool', net['conv3_4'])
    
        net['conv4_1'] = self.build_net('conv', net['pool3'], self.get_weight_bias(vgg_layers, 19))
        net['conv4_2'] = self.build_net('conv', net['conv4_1'], self.get_weight_bias(vgg_layers, 21))
        net['conv4_3'] = self.build_net('conv', net['conv4_2'], self.get_weight_bias(vgg_layers, 23))
        net['conv4_4'] = self.build_net('conv', net['conv4_3'], self.get_weight_bias(vgg_layers, 25))
        net['pool4'] = self.build_net('pool', net['conv4_4'])
    
        net['conv5_1'] = self.build_net('conv', net['pool4'], self.get_weight_bias(vgg_layers, 28))
        net['conv5_2'] = self.build_net('conv', net['conv5_1'], self.get_weight_bias(vgg_layers, 30))
        net['conv5_3'] = self.build_net('conv', net['conv5_2'], self.get_weight_bias(vgg_layers, 32))
        net['conv5_4'] = self.build_net('conv', net['conv5_3'], self.get_weight_bias(vgg_layers, 34))
        net['pool5'] = self.build_net('pool', net['conv5_4'])
        self.net = net
        self.input = net["input"]

vgg = Vgg19Model(vgg_mat_file) 

def load_img(src):
    img_obj = Image.open(src)
    img_obj = img_obj.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
    img_data = np.array(list(img_obj.getdata()))
    img_data = img_data[...,:3]
    img_data = img_data.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    return img_data - vgg.MEAN_VALUES

content_img = load_img(content_file)
style_img = load_img(style_file)

# =====================

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable init 只是为了算 content_fms 与 style_fms

content_fms = sess.run([vgg.net[fm_name] for fm_name in content_fm_names], feed_dict={vgg.input: content_img})
style_fms = sess.run([vgg.net[fm_name] for fm_name in style_fm_names], feed_dict={vgg.input: style_img})

# -- loss function

content_loss = 0.0
for (i, fm_name) in enumerate(content_fm_names):
    diff = content_fms[i] - vgg.net[fm_name]
    content_loss += tf.reduce_sum(tf.pow(diff, 2))
content_loss = content_loss / 2.

style_loss = 0.0
for (i, fm_name) in enumerate(style_fm_names):
    t1 = tf.reshape(style_fms[i], (-1, tf.shape(style_fms[i])[-1]))
    t2 = tf.reshape(vgg.net[fm_name], (-1, tf.shape(vgg.net[fm_name])[-1]))
    diff = tf.matmul(tf.transpose(t1), t1) - tf.matmul(tf.transpose(t2), t2)
    style_loss += tf.reduce_sum(tf.pow(diff, 2))
style_loss = style_loss / ((2. * tf.cast(tf.size(style_fms[i]), tf.float32))**2)
style_loss /= len(style_fms)

total_loss = content_loss_weight * content_loss + style_loss_weight * style_loss

optimizer = tf.train.AdamOptimizer(learn_rate)
opt = optimizer.minimize(total_loss)

sess.run(tf.global_variables_initializer()) # 这才开始初始化模型变量

if noise_img_use_content_img:
    # 待训练优化的图片的初始化，是否是 content image 加噪声得到
    # 这样会加速训练。如果待训练优化的图片的初始化完全用随机初始化，也能得到结果，但是会慢些
    sess.run(tf.assign(vgg.input, vgg.generate_noise_image(content_img)))

# -- train 
for i in xrange(step_cnt):
    _, loss = sess.run([opt, total_loss])
    if i % save_every_cnt == 0:
        out_img = sess.run(tf.clip_by_value(tf.cast(vgg.input, tf.int32) + vgg.MEAN_VALUES, 0, 255))
        out_img = np.reshape(out_img, (-1, 3))
        out_img = [tuple(j) for j in out_img]
        img_obj = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white") 
        img_obj.putdata(out_img, 1., 0.)
        img_obj.save("%s/%d.png" % (save_path, i))
        print "step=%d save generated image to %s/%d.png" % (i, save_path, i)
    print "step=%d loss=%.6f" % (i, loss)

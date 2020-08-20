# -*- coding:UTF-8 -*-
import numpy as np
import tensorflow as tf

import pandas as pd
# tensorflow 版本要低
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
# download and load Mnist library(55000 * 28*28_

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 代表训练数据
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.

# 代表的是10个标签，0,1,2,3....9
output_y = tf.placeholder(tf.int32, [None, 10])

# shape 前面使用-1,能够自动的对其形状进行推导
input_x_image = tf.reshape(input_x, [-1, 28, 28, 1])

test_x = mnist.test.images[:3000]  # picture
test_y = mnist.test.labels[:3000]  # label

# 创建 CNN 模型
# 构建第一层 CNN 模型,卷积层
cover1 = tf.layers.conv2d(
	inputs = input_x_image,  # shape is [28, 28, 1]
	filters = 32,            # 设置卷积深度为32,意思也就是说有32个卷积核
	kernel_size = [5, 5],    # 设置卷积核的大小
	strides = 1,             # 设置卷积的步长
	padding = "same",        # 进行卷积后,大小不变
	activation = tf.nn.relu  # 使用 Relu 这个激活函数
)   # [28, 28, 32]

# 构建第一层池化层,作用是对第一层卷积后的结果进行降维,获得池化大小区域内的单个数据进行填充
pool1 = tf.layers.max_pooling2d(
	inputs=cover1,          # shape [28,28,32]
	pool_size=[2, 2],       # 设置池化层的大小
	strides=2               # 设置池化层的步长
)    # shape [14, 14, 32]

# 构建第二层 CNN 模型,卷积层
cover2 = tf.layers.conv2d(
	inputs=pool1,          # shape is [14, 14, 32]
	filters=64,            # 采用64个卷积核
	kernel_size=[5, 5],    # 设置卷积核的大小
	strides=1,             # 设置卷积的步长
	padding="same",        # 进行卷积后,大小不变
	activation=tf.nn.relu  # 使用 Relu 这个激活函数
)   # shape [14, 14, 64]

# 构建第二层池化层,作用是对第二层卷积后的结果进行降维,获得池化大小区域内的单个数据进行填充
pool2 = tf.layers.max_pooling2d(
	inputs=cover2,          # shape [14,14,64]
	pool_size=[2, 2],       # 设置池化层的大小
	strides=2               # 设置池化层的步长
)    # shape [7, 7, 64]

# 展开第二层池化后的数据,使得其维度为一维数组
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # shape [7*7*64]

# 设置全连接层网络,共有 1024 个神经元,并且采用Relu这个激活函数
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# 为了避免1024个全连接网络神经元出现过拟合,采用Dropout丢弃掉一半的连接,即rate = 0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=True)

# 定义最后输出10个节点,因为是0-9的数字,一共10个
logites = tf.layers.dense(inputs=dropout, units=10)  # shape [1*1*10]

# 通过使用 softmax 对所有的预测结果和正确结果进行比较并计算概率,
# 然后再使用交叉熵计算概率密度误差,也就是我们的损失函数
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logites)

# 采用 Adam 优化器去优化误差,设置学习率为0.001,能够更好的进行优化
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 计算正确率,正确率的计算步骤:
# 1、对所有的待检测数据进行识别并与正确的结果进行判断,返回bool类型;
# 2、将所有的bool结果进行float操作然后求均值,这个均值就是正确率;
# tf.metrics.accuracy() will return (accuracy,update_op)
accuracy = tf.metrics.accuracy(
	labels=tf.argmax(output_y, axis=1),  # 正确的数字(label)
	predictions=tf.argmax(logites, axis=1)  # 预测的数字(label)
)[1]

with tf.Session() as sess:
    # 初始化局部和全局变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # 保存tensor图
    tf.summary.FileWriter('./log', sess.graph)

    # 定义一共训练5000次
    for i in range(5000):
        # 每次的数据从mnist训练数据集中选取 50 份出来训练
        batch = mnist.train.next_batch(50)  # get 50 sample

        train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})

        # 每训练100次打印一次训练模型的识别率
        if i % 100 == 0:
            test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
            print('Step=%d, Train loss=%.6f, [Test accuracy=%.6f]' % (i, train_loss, test_accuracy))

    # 最后一次测试:从测试数据集中选取前 20 张图片进行识别
    # 1.利用现在的模型进行预测数字,test_output 形状是[20,10]
    test_output = sess.run(logites, {input_x: test_x[:20]})
    # 2.获取最大可能性的数字,一维直接返回具体值,二维以上返回下标索引
    inferenced = np.argmax(test_output, 1)
    # 3.打印预测的数字和实际对应的数字
    print('inferenced number:')
    print(inferenced)
    print('Real number:')
    print(np.argmax(test_y[:20], 1))








# -*- coding: utf-8 -*-
"""
 @Time    : 2019-08-21 14:27
 @Author  : sishuyong
 @File    : drop_out.py
 @brief   : 
"""
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def test_digits():
    def add_layer(input, input_size, output_size, layer_name, activation_function=None):
        Weights = tf.Variable(tf.random_normal([input_size, output_size]))
        biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        Wx_plus_bias = tf.add(tf.matmul(input, Weights), biases)
        Wx_plus_bias = tf.nn.dropout(Wx_plus_bias, keep_prob)

        if activation_function == None:
            outputs = Wx_plus_bias
        else:
            outputs = activation_function(Wx_plus_bias)

        # 这里的output是一个二维的，所以每一步对应一个线（或者说小的矩形，颜色越深的地方表示这个地方的数越多，可以认为纵向上表示train到这一步的时候的一个数据分布
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs

    digits = load_digits()
    X = digits.data
    Y = digits.target
    Y = LabelBinarizer().fit_transform(Y)
    print(Y.shape)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)

    xs = tf.placeholder(tf.float32, [None, 64])
    ys = tf.placeholder(tf.float32, [None, 10])

    keep_prob = tf.placeholder(tf.float32)

    l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
    prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

    # 因为cross_entropy是一个标量，所以定义tf.summary.scalar
    tf.summary.scalar("loss", cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 合并所有的summary
        merged = tf.summary.merge_all()
        # 得到summary的FileWriter
        train_writer = tf.summary.FileWriter("logs/train/", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test/", sess.graph)

        sess.run(init)

        for i in range(1000):
            sess.run(train_step, feed_dict={xs: train_x, ys: train_y, keep_prob: 0.5})

            if i % 50 == 0:
                cur_loss = sess.run(cross_entropy, feed_dict={xs: train_x, ys: train_y, keep_prob: 0.5})
                print(cur_loss)

                train_loss = sess.run(merged, feed_dict={xs: train_x, ys: train_y, keep_prob: 0.5})
                test_loss = sess.run(merged, feed_dict={xs: test_x, ys: test_y, keep_prob: 0.5})

                # 将loss写入FileWriter中
                train_writer.add_summary(train_loss, i)
                test_writer.add_summary(test_loss, i)


def test_reduce_mean_sum():
    import numpy as np
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(x)

    with tf.Session() as sess:
        print(sess.run(x))
        print(sess.run(tf.reduce_mean(x)))
        print(sess.run(tf.reduce_mean(x)))

        print(sess.run(tf.reduce_mean(x, 0)))

        print(sess.run(tf.reduce_mean(x, 1)))

        print('reduce sum....')
        print(sess.run(tf.reduce_sum(x)))
        print(sess.run(tf.reduce_sum(x, axis=[0, 1], keepdims=True)))
        print(sess.run(tf.reduce_sum(x, axis=[0, 1], keepdims=True)))

        print(sess.run(tf.reduce_sum(x, 0, keepdims=True)))
        print(sess.run(tf.reduce_sum(x, 0, keepdims=False)))

        print(sess.run(tf.reduce_sum(x, 1, keepdims=True)))

        print(sess.run(tf.reduce_sum(x, axis=[1])))


if __name__ == "__main__":
    # test_digits()
    test_reduce_mean_sum()

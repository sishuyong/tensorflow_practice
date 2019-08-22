# -*- coding: utf-8 -*-
"""
 @Time    : 2019-08-22 11:25
 @Author  : sishuyong
 @File    : embedding_basic.py
 @brief   :
"""

import tensorflow as tf
import numpy as np


def em_test():
    # embedding lookup
    embedding = tf.constant(
        [
            [0.21, 0.41, 0.51, 0.11],
            [0.22, 0.42, 0.52, 0.12],
            [0.23, 0.43, 0.53, 0.13],
            [0.24, 0.44, 0.54, 0.14]
        ],
        dtype=tf.float32
    )

    feature_batch = tf.constant([2, 3, 1, 0])
    get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)

    feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)

    get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)

    with tf.Session() as sess:
        print("embedding lookup 1 ------------------------")
        sess.run(tf.global_variables_initializer())
        embedding1, embedding2, one_hot_feat = sess.run([get_embedding1, get_embedding2, feature_batch_one_hot])
        print(embedding1)
        print(embedding2)
        print(one_hot_feat)
        print("#" * 50)


def t_em_2():
    # embedding lookup
    embedding = tf.get_variable(name="embedding", shape=[4, 4], dtype=tf.float32,
                                initializer=tf.random_uniform_initializer)
    feature_batch = tf.constant([2, 3, 1, 0])
    get_embedding_1 = tf.nn.embedding_lookup(embedding, feature_batch)

    feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)
    get_embedding_2 = tf.matmul(feature_batch_one_hot, embedding)

    with tf.Session() as sess:
        print("embedding lookup 2 ------------------------")
        sess.run(tf.global_variables_initializer())
        embedding1, embedding2 = sess.run([get_embedding_1, get_embedding_2])
        print(embedding1)
        print(embedding2)


def get_gather():
    # 单维索引
    embedding = tf.constant(
        [
            [0.21, 0.41, 0.51, 0.11],
            [0.22, 0.42, 0.52, 0.12],
            [0.23, 0.43, 0.53, 0.13],
            [0.24, 0.44, 0.54, 0.14]
        ], dtype=tf.float32)
    index_a = tf.Variable([2, 3, 1, 0])
    gather_a = tf.gather(embedding, index_a)

    gather_a_axis1 = tf.gather(embedding, index_a, axis=1)

    b = tf.Variable(np.arange(10))
    index_b = tf.Variable([2, 4, 6, 8])
    gather_b = tf.gather(b, index_b)
    with tf.Session() as sess:
        print("embedding gather ------------------------")
        sess.run(tf.global_variables_initializer())

        print(sess.run(gather_a))
        print(sess.run(gather_a_axis1))
        print(sess.run(gather_b))


def sparse_embedding():
    # sparse embedding
    idx = tf.SparseTensor(indices=[[0, 0], [1, 2], [1, 3]], values=[1, 2, 1], dense_shape=[2, 4])
    b = tf.sparse_tensor_to_dense(idx, default_value=-1)

    embedding = tf.constant(
        [
            [0.21, 0.41, 0.51, 0.11],
            [0.22, 0.42, 0.52, 0.12],
            [0.23, 0.43, 0.53, 0.13],
            [0.24, 0.44, 0.54, 0.14]
        ], dtype=tf.float32)
    embedding_sparse = tf.nn.embedding_lookup_sparse(embedding, sp_ids=idx, sp_weights=None, combiner="sum")

    """
    embedding_sparse shape=(idx.shape[0], embedding.shape[1])
    idx dense结果如下：
    [[ 1 -1 -1 -1]
    [-1 -1  2  1]]
    embedding_sparse 结果如下：
    [[0.22 0.42 0.52 0.12]
    [0.45 0.85 1.05 0.25]]
    embedding_sparse 的第一行分析： 
        idx 第一行中非-1 的值"1"，"1" 为embedding 中的行。由于只有一个非-1 的值，
        embedding_sparse的第一行等于embeding的第"1"行
    embedding_sparse 的第二行分析：
        idx中第二行非-1 的值"2"，"1"。 
        "2" 与 "1" 是embedding中的行索引，
        embedding_sparse[1] = embedding[2] + embedding[1]
        （此处的加法是由于 combiner="sum"）
    """

    with tf.Session() as sess:
        print("sparse embedding--------------------------")
        sess.run(tf.global_variables_initializer())
        print(sess.run(b))
        print(sess.run(embedding_sparse))


if __name__ == "__main__":
    # em_test()
    # t_em_2()
    # get_gather()
    sparse_embedding()

# -*- coding: utf-8 -*-
"""
 @Time    : 2019-08-22 11:25
 @Author  : sishuyong
 @File    : embedding_test.py
 @brief   : 
"""

import tensorflow as tf

# embedding
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
    sess.run(tf.global_variables_initializer())
    embedding1, embedding2, one_hot_feat = sess.run([get_embedding1, get_embedding2, feature_batch_one_hot])
    print(embedding1)
    print(embedding2)
    print(one_hot_feat)

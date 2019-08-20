# -*- coding: utf-8 -*-
"""
 @Time    : 2019-08-20 17:15
 @Author  : sishuyong
 @File    : basic.py
 @brief   : 
"""

"""
TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图,
然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算. 节点（Nodes）在图
中表示数学操作,图中的线（edges）则表示在节点间相互联系的多维数据数组, 即张量（tensor).
训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来.

Tensor 张量意义
张量（Tensor):
张量有多种. 零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 [1]
一阶张量为 向量 (vector), 比如 一维的 [1, 2, 3]
二阶张量为 矩阵 (matrix), 比如 二维的 [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
以此类推, 还有 三阶 三维的 …
"""

"""先来看一个例子"""

import tensorflow as tf
import numpy as np

# tensorflow中大部分数据是float32
# create real data
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3

# create tensorflow structure start
# 定义变量



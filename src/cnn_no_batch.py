#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""author : haozhuolin
   date : 20170728
"""

import sys
import gzip
import cPickle

import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


class Convolution(object):
    def __init__(self, conv_size, step, zero_padding=0):
        self.conv_num = conv_size[0]
        self.conv_depth = conv_size[1]
        self.conv_length = conv_size[2]
        self.conv_width = conv_size[3]
        self.weight = np.random.normal(0, 1,
                                       (self.conv_num, self.conv_depth,
                                        self.conv_length, self.conv_width))
        self.bias = np.random.normal(0, 1, (self.conv_num, 1))
        self.step = step
        self.zero_padding = zero_padding

    def relu(self, z):
        z[np.where(z < 0)] = 0
        return z

    def padding(self, x):
        x = np.array([np.pad(xx, self.zero_padding, 'constant') for xx in x])
        return x

    def forward(self, x):
        # 补0
        x = self.padding(x)
        # 计算卷积后的矩阵长宽
        new_length = 1.0 * (x.shape[1] - self.conv_length) / self.step + 1
        new_width = 1.0 * (x.shape[2] - self.conv_width) / self.step + 1
        if int(new_length) != new_length:
            raise Exception(
                "error (row: %s - conv_width: %s) / step: %s is not integer" %
                (x.shape[1], self.conv_length, self.step))
        if int(new_width) != new_width:
            raise Exception(
                "error (col: %s - conv_width: %s) / step: %s is not integer" %
                (x.shape[2], self.conv_width, self.step))
        new_length = int(new_length)
        new_width = int(new_width)
        result = np.zeros((self.conv_num, new_length, new_width))
        # weight为一个卷积核
        for w, weight, bias in zip(
                range(self.conv_num), self.weight, self.bias):
            for i in range(new_length):
                for j in range(new_width):
                    val = np.sum(
                        x[:, self.step * i:self.step * i + self.conv_length,
                          self.step * j:self.step * j + self.conv_width] *
                        weight) + bias
                    result[w, i, j] = val
        return result, self.relu(result)


class Pooling(object):
    def __init__(self, pooling_width, pooling_depth):
        self.pooling_width = pooling_width
        self.pooling_depth = pooling_depth

    def forward(self, x):
        new_length = 1.0 * x.shape[1] / self.pooling_width
        new_width = 1.0 * x.shape[2] / self.pooling_width
        if int(new_length) != new_length:
            raise Exception(
                "error row: %s / pooling_width: %s is not integer" %
                (x.shape[1], self.pooling_width))
        if int(new_width) != new_width:
            raise Exception(
                "error col: %s / pooling_width: %s is not integer" %
                (x.shape[2], self.pooling_width))
        new_length = int(new_length)
        new_width = int(new_width)
        result = np.zeros((self.pooling_depth, new_length, new_width))
        for i in range(new_length):
            for j in range(new_width):
                val = np.max(x[:, i * self.pooling_width:(
                    i + 1) * self.pooling_width, j * self.pooling_width:(
                        j + 1) * self.pooling_width], (1, 2))
                result[:, i, j] = val
        return result, result


class Full(object):
    def __init__(self, full_in, full_out):
        self.weight = np.random.normal(0, 1,
                                       (full_out, full_in)) / np.sqrt(full_in)
        self.bias = np.random.normal(0, 1, (full_out, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        :param z: z
        :return: the derivative of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, x):
        z = self.weight.dot(x.flatten().reshape(-1, 1)) + self.bias
        return z, self.sigmoid(z)


class NetWork(object):
    def __init__(self, eta=0.3, iter_num=300, loss="quadratic"):
        self.eta = eta
        self.iter_num = iter_num
        self.loss = loss
        self.layer = []

    def add_convolution(self,
                        conv_num,
                        conv_length,
                        conv_width,
                        step,
                        zero_padding=0):
        conv_depth = 1
        if self.layer:
            if isinstance(self.layer[-1], Convolution):
                conv_depth = self.layer[-1].conv_num
            elif isinstance(self.layer[-1], Pooling):
                conv_depth = self.layer[-1].pooling_depth

        self.layer.append(
            Convolution([conv_num, conv_depth, conv_length, conv_width], step,
                        zero_padding))

    def add_pooling(self, pooling_width):
        pooling_depth = 1
        if self.layer:
            if isinstance(self.layer[-1], Convolution):
                pooling_depth = self.layer[-1].conv_num
            elif isinstance(self.layer[-1], Pooling):
                pooling_depth = self.layer[-1].pooling_depth
        self.layer.append(Pooling(pooling_width, pooling_depth))

    def add_full(self, full_in, full_out):
        self.layer.append(Full(full_in, full_out))

    def forward_propagation(self, x):
        z = []
        a = [x]
        for layer in self.layer:
            zi, ai = layer.forward(a[-1])
            z.append(zi)
            a.append(ai)
        return z, a

    def back_propagation(self, x, y):
        z, a = self.forward_propagation(x)
        for i in a:
            print i.shape

    def fit(self, train_data, test_data=None):
        X, Y = train_data
        for i in range(self.iter_num):
            for x, y in zip(X, Y):
                self.back_propagation(x, y)
                break
            break


def vectorized_y(i):
    e = np.zeros((10))
    e[i] = 1.0
    return e


def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = cPickle.load(f)
    f.close()
    train_data = [
        train_data[0].reshape(train_data[0].shape[0], 1, 28, 28),
        np.array([vectorized_y(i) for i in train_data[1]])
    ]
    val_data = [
        val_data[0].reshape(val_data[0].shape[0], 1, 28, 28),
        np.array([vectorized_y(i) for i in val_data[1]])
    ]
    test_data = [
        test_data[0].reshape(test_data[0].shape[0], 1, 28, 28),
        np.array([vectorized_y(i) for i in test_data[1]])
    ]
    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = load_data()
    network = NetWork(eta=0.3, iter_num=1, loss="quadratic")
    network.add_convolution(
        conv_num=3, conv_length=3, conv_width=3, step=1, zero_padding=1)
    network.add_convolution(
        conv_num=5, conv_length=3, conv_width=3, step=1, zero_padding=0)
    network.add_pooling(pooling_width=2)
    network.add_convolution(
        conv_num=7, conv_length=3, conv_width=3, step=1, zero_padding=0)
    network.add_convolution(
        conv_num=9, conv_length=3, conv_width=3, step=1, zero_padding=0)
    network.add_pooling(pooling_width=3)
    network.add_full(full_in=81, full_out=10)
    network.fit(train_data)

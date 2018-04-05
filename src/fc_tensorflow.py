#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""author : haozhuolin
   date : 20170728
"""

import sys
import gzip
import numpy as np
import cPickle
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf-8")


class NetWorkTensorflow(object):
    def __init__(self,
                 layer,
                 eta=0.3,
                 min_batch=100,
                 iter_num=300,
                 loss="quadratic"):
        """
        :param layer: structure of neural networks
        :param eta: learning rate
        :param min_batch: the size of each batch
        :param iter_num: the number of iterations
        :param loss: the type of loss function
        """
        self.eta = eta
        self.min_batch = min_batch
        self.iter_num = iter_num
        self.loss = loss
        self.layer_num = len(layer)
        # It should be noted here, initialization is very important
        # self.weights = [tf.Variable(tf.random_normal([layer[l], layer[l - 1]])) for l in
        #                 range(1, self.layer_num)]
        self.weights = [
            tf.Variable(
                tf.random_normal([layer[l], layer[l - 1]]) / tf.sqrt(
                    1.0 * layer[l - 1])) for l in range(1, self.layer_num)
        ]
        self.bias = [
            tf.Variable(tf.zeros([layer[l], 1]))
            for l in range(1, self.layer_num)
        ]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward_propagation(self, a):
        """
        :param a: the feature matrix
        :return: the forward propagation function
        """
        for w, b in zip(self.weights, self.bias):
            a = tf.nn.sigmoid(tf.matmul(w, a) + b)
        return a

    def cost(self, yf, yp):
        """
        :param yf: the output value of the model
        :param yp: actual value
        :return: the cost function
        """
        if self.loss == "quadratic":
            tmp_loss = tf.square(yf - yp)
        elif self.loss == "crossEntropy":
            tmp_loss = -1 * (yp * tf.log(yf) + (1 - yp) * tf.log(1 - yf))
        return tf.reduce_mean(tf.reduce_sum(tmp_loss, reduction_indices=0))

    def fit(self, X, y, test_data):
        """
        :param X: the feature matrix of training data
        :param y: the actual value of training data
        :param test_data: the test data
        :return: fitting function
        """
        xp = tf.placeholder(tf.float32, shape=[784, None])
        yp = tf.placeholder(tf.float32, shape=[10, None])
        yf = self.forward_propagation(xp)
        loss = self.cost(yf, yp)
        optimizer = tf.train.GradientDescentOptimizer(self.eta)
        train = optimizer.minimize(loss, var_list=[self.weights, self.bias])
        correct_prediction = tf.equal(tf.argmax(yp, 0), tf.argmax(yf, 0))
        for i in xrange(self.iter_num):
            for k in xrange(0, X.shape[0], self.min_batch):
                batch_X = X[k:k + self.min_batch, ]
                batch_y = y[k:k + self.min_batch]
                self.sess.run(train, feed_dict={xp: batch_X.T, yp: batch_y.T})

            print self.sess.run(loss, feed_dict={xp: X.T, yp: y.T}), \
                np.mean(self.sess.run(correct_prediction, feed_dict={xp: test_data[0].T, yp: test_data[1].T}))


def vectorized_y(i):
    """
    :param i: digital
    :return: 10-dimensional vector, except the i-th dimension is 1, the other dimensions are 0
    """
    e = np.zeros((10))
    e[i] = 1.0
    return e


def load_data():
    """
    :return: load data function
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = cPickle.load(f)
    f.close()
    train_data = [
        train_data[0],
        np.array([vectorized_y(i) for i in train_data[1]])
    ]
    val_data = [val_data[0], np.array([vectorized_y(i) for i in val_data[1]])]
    test_data = [
        test_data[0],
        np.array([vectorized_y(i) for i in test_data[1]])
    ]
    return train_data, val_data, test_data


if __name__ == "__main__":
    """main
    """
    train_data, val_data, test_data = load_data()
    network = NetWorkTensorflow(
        [784, 100, 10],
        eta=0.03,
        min_batch=64,
        iter_num=100,
        loss="crossEntropy")
    network.fit(train_data[0], train_data[1], test_data)

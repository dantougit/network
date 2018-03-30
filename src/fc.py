#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""author : dantou
   date : 20170728
"""

import sys
import gzip
import numpy as np
import cPickle

reload(sys)
sys.setdefaultencoding("utf-8")


class NetWork(object):
    def __init__(self, layer, eta=0.3, min_batch=100, iter_num=300, loss="quadratic"):
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
        # self.weights = [np.random.normal(0, 1, (layer[l], layer[l - 1])) for l in
        #                 range(1, self.layer_num)]
        self.weights = [np.random.normal(0, 1, (layer[l], layer[l - 1])) / np.sqrt(layer[l - 1]) for l in
                        range(1, self.layer_num)]
        self.bias = [np.zeros((layer[l], 1)) for l in range(1, self.layer_num)]

    def sigmoid(self, z):
        """
        :param z: z
        :return: sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        :param z: z
        :return: the derivative of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, a, y):
        """
        :param a: the output value of the model
        :param y: actual value
        :return: the derivative of the cost function
        """
        if self.loss == "quadratic":
            return a - y
        elif self.loss == "crossEntropy":
            return (a - y) / (a * (1 - a))

    def forward_propagation(self, X):
        """
        :param X: the feature matrix
        :return: the forward propagation function
        """
        z = []
        a = [X]
        for w, b in zip(self.weights, self.bias):
            zi = w.dot(a[-1]) + b
            z.append(zi)
            a.append(self.sigmoid(zi))
        return z, a

    def back_propagation(self, X, y):
        """
        :param X: the feature matrix
        :param y: the actual value
        :return: the back propagation function
        """
        z, a = self.forward_propagation(X)
        delta = [self.cost_derivative(a[-1], y) * self.sigmoid_derivative(z[-1])]
        for l in range(1, self.layer_num - 1)[::-1]:
            delta.insert(0, self.weights[l].T.dot(delta[0]) * self.sigmoid_derivative(z[l - 1]))
        for l in range(self.layer_num - 1):
            self.weights[l] -= self.eta / X.shape[1] * delta[l].dot(a[l].T)
            self.bias[l] -= self.eta * np.mean(delta[l], 1).reshape(-1, 1)

    def fit(self, train_data, test_data=None):
        """
        :param train_data: the training data
        :param test_data: the test data
        :return: fitting function
        """
        X, y = train_data
        for i in range(self.iter_num):
            for k in xrange(0, X.shape[0], self.min_batch):
                batch_X = X[k:k + self.min_batch, ]
                batch_y = y[k:k + self.min_batch]
                self.back_propagation(batch_X.T, batch_y.T)
            if test_data:
                print self.evaluate(X.T, y.T), self.evaluate(test_data[0].T, test_data[1].T)

    def evaluate(self, test_X, test_y):
        """
        :param test_X: feature matrix
        :param test_y: actual value
        :return: evaluation function
        """
        z, a = self.forward_propagation(test_X)
        return 1.0 * np.sum(a[-1].argmax(0) == test_y.argmax(0)) / test_X.shape[1]


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
    train_data = [train_data[0], np.array([vectorized_y(i) for i in train_data[1]])]
    val_data = [val_data[0], np.array([vectorized_y(i) for i in val_data[1]])]
    test_data = [test_data[0], np.array([vectorized_y(i) for i in test_data[1]])]
    return train_data, val_data, test_data


if __name__ == "__main__":
    """main
    """
    train_data, val_data, test_data = load_data()
    network = NetWork([784, 64, 10], eta=0.1, min_batch=64, iter_num=100, loss="crossEntropy")
    network.fit(train_data, test_data)

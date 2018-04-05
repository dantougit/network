#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""author : haozhuolin
   date : 20170728
"""

import sys
import gzip
import numpy as np
import cPickle
from keras.layers import Input, Dense
from keras.models import Model

reload(sys)
sys.setdefaultencoding("utf-8")


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

    inputs = Input(shape=(784, ))
    x = Dense(128, activation='relu')(inputs)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=64,
        epochs=10,
        validation_split=None)

    classes = model.predict(test_data[0])
    print 1.0 * np.sum(
        classes.argmax(1) == test_data[1].argmax(1)) / test_data[1].shape[0]

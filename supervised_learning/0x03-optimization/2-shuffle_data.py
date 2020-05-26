#!/usr/bin/env python3
"""Contains the shuffle_data function"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    :param X: first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    :param Y: second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    :return: the shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]

#!/usr/bin/env python3
"""Contains the normalization_constants function"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix
    :param X: numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    :return: the mean and standard deviation of each feature
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s

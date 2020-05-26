#!/usr/bin/env python3
"""Contains the normalize function"""


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix
    :param X: numpy.ndarray of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    :param m: numpy.ndarray of shape (nx,)
        that contains the mean of all features of X
    :param s: numpy.ndarray of shape (nx,)
        that contains the standard deviation of all features of X
    :return: The normalized X matrix
    """
    Z = (X - m) / s
    return Z

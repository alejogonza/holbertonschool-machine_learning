#!/usr/bin/env python3
"""Contains the one_hot_encode method"""

import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    Returns: a numpy.ndarray with shape (m, )
        containing the numeric labels for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if not np.where((one_hot == 0) | (one_hot == 1), True, False).all():
        return None
    classes, m = one_hot.shape
    if np.sum(one_hot) != m:
        return None

    Y = np.zeros(m)
    tmp = np.arange(m)

    "index of max in column i corresponding to value of sample i "
    axis = np.argmax(one_hot, axis=0)
    Y[tmp] = axis
    return Y.astype("int64")

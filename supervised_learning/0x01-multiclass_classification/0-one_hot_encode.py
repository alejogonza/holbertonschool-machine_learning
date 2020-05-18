#!/usr/bin/env python3
"""Contains the one_hot_encode method"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    Y is a numpy.ndarray with shape (m,) containing numeric class labels
        m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m),
        or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    m = len(Y)
    a = np.arange(m)
    one_hot = np.zeros((classes, m))
    one_hot[Y, a] = 1

    return one_hot

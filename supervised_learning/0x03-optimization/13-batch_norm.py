#!/usr/bin/env python3
"""Contains the batch_norm function"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using
    batch normalization
    :param Z: numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
    :param gamma: numpy.ndarray of shape (1, n)
        containing the scales used for batch normalization
    :param beta: numpy.ndarray of shape (1, n) containing
        the offsets used for batch normalization
    :param epsilon: small number used to avoid division by zero
    :return: normalized Z matrix
    """
    m = np.mean(Z, axis=0)
    s = np.var(Z, axis=0)

    # normalization step
    Z_norm = (Z - m) / ((s + epsilon)**(1/2))

    # introduction of trainable parameters gamma for scale and beta for offset
    # allows to take advantage of a non strictly normalized distribution
    # non zero mean (offset) and non one stv (scale)
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde

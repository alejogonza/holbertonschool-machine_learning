#!/usr/bin/env python3
"""Contains the dropout_forward_prop function"""

import numpy as np
import tensorflow as tf

def dropout_forward_prop(X, weights, L, keep_prob):
    """

    :param X: numpy.ndarray of shape (nx, m) containing the input data for the network
        nx is the number of input features
        m is the number of data pointsnumpy.ndarray of shape (nx, m) containing the input data for the network
    :param weights: dictionary of the weights and biases of the neural network
    :param L:
    :param keep_prob: probability that a node will be kept
    :return: dictionary containing the outputs of each layer and the dropout
        mask used on each layer (see example for format)
    """
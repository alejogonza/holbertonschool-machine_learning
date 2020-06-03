#!/usr/bin/env python3
"""Contains the l2_reg_gradient_descent function"""

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    pdates the weights and biases of a neural network using
        gradient descent with L2 regularization
    :param Y: one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
    :param weights: a dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs of each layer of the neural network
    :param alpha: learning rate
    :param lambtha: L2 regularization parameter
    :param L: number of layers of the network
    The weights and biases of the network should be updated in place
    """
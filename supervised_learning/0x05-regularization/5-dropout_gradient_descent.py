#!/usr/bin/env python3
"""Contains the dropout_gradient_descent function"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent
    :param Y: one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs and dropout masks of each layer of the neural network
    :param alpha: learning rate
    :param keep_prob: probability that a node will be kept
    :param L: number of layers of the network
    The weights of the network should be updated in place
    """
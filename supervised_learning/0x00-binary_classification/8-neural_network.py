#!/usr/bin/env python3
"""Contains the NeuralNetwork"""

import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class
    defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        constructor
        :param nx: number of input features
        :param nodes: number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        """The weights vector for the hidden layer. Upon instantiation,
            it should be initialized using a random normal distribution"""
        self.W1 = np.random.normal(0, 1, (nodes, nx))

        """The bias for the hidden layer. Upon instantiation,
            it should be initialized with 0’s"""
        self.b1 = np.zeros((nodes, 1))

        """The activated output for the hidden layer. Upon instantiation,
            it should be initialized to 0"""
        self.A1 = 0

        """The weights vector for the output neuron. Upon instantiation,
            it should be initialized using a random normal distribution"""
        self.W2 = np.random.normal(0, 1, (1, nodes))

        """The bias for the output neuron. Upon instantiation,
            it should be initialized to 0"""
        self.b2 = 0

        """The activated output for the output neuron (prediction).
            Upon instantiation, it should be initialized to 0"""
        self.A2 = 0

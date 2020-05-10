#!/usr/bin/env python3
"""Contains the Neuron class"""

import numpy as np


# Miscellaneous functions
def sigmoid(Z):
    """sigmoid function"""
    return 1.0 / (1.0 + np.exp(-Z))


class Neuron:
    """Neuron class
    nx is the number of input features to the neuron
    """

    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        """The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution."""
        self.__W = np.random.normal(0, 1, (1, nx))

        """The bias for the neuron. Upon instantiation,
        it should be initialized to 0."""
        self.__b = 0

        """The activated output of the neuron (prediction). Upon instantiation,
        it should be initialized to 0."""
        self.__A = 0

    @property
    def W(self):
        """property to retrieve it"""
        return self.__W

    @property
    def b(self):
        """property to retrieve it"""
        return self.__b

    @property
    def A(self):
        """property to retrieve it"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        """
        Z = np.matmul(self.__W, X) + self.__b
        """self.__A = 1 / (1 + np.exp(-Z))"""
        self.__A = sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data
        A is a numpy.ndarray with shape (1, m)
        containing the activated output of the neuron for each example
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data
        """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        :param X: a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :param Y: a numpy.ndarray with shape (1, m)
        :param A: a numpy.ndarray with shape (1, m)
        :param alpha: the learning rate
        Updates the private attri butes __W and __b
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - alpha * db

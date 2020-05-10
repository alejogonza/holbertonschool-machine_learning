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
        self.__W1 = np.random.normal(0, 1, (nodes, nx))

        """The bias for the hidden layer. Upon instantiation,
            it should be initialized with 0’s"""
        self.__b1 = np.zeros((nodes, 1))

        """The activated output for the hidden layer. Upon instantiation,
            it should be initialized to 0"""
        self.__A1 = 0

        """The weights vector for the output neuron. Upon instantiation,
            it should be initialized using a random normal distribution"""
        self.__W2 = np.random.normal(0, 1, (1, nodes))

        """The bias for the output neuron. Upon instantiation,
            it should be initialized to 0"""
        self.__b2 = 0

        """The activated output for the output neuron (prediction).
            Upon instantiation, it should be initialized to 0"""
        self.__A2 = 0

    @property
    def W1(self):
        """property to retrieve W1"""
        return self.__W1

    @property
    def b1(self):
        """property to retrieve b1"""
        return self.__b1

    @property
    def A1(self):
        """property to retrieve A1"""
        return self.__A1

    @property
    def W2(self):
        """property to retrieve W2"""
        return self.__W2

    @property
    def b2(self):
        """property to retrieve b2"""
        return self.__b2

    @property
    def A2(self):
        """property to retrieve A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X:  is a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :return: private attributes __A1 and __A2, respectively
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        :param A: a numpy.ndarray with shape (1, m)
            containing the activated output of the neuron for each example
        :return: the cost
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        :param X: a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :param Y:  is a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        :return: the neuron’s prediction and the cost of the network
        """
        self.forward_prop(X)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        :param X: a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :param Y: a numpy.ndarray with shape (1, m)
        that contains the correct labels for the input data
        :param A1: the output of the hidden layer
        :param A2: the predicted output
        :param alpha: the learning rate
        """
        m = A1.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1a = np.matmul(self.__W2.T, dZ2)
        dZ1b = A1 * (1 - A1)
        dZ1 = dZ1a * dZ1b
        dW1 = np.matmul(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - alpha * db2

        self.__W1 = self.__W1 - (alpha * dW1).T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        :param self:
        :param X: a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :param Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)

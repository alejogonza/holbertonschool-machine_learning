#!/usr/bin/env python3
"""Contains the create_train_op function"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network:
    :param loss: loss of the network’s prediction
    :param alpha: learning rate
    :return: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)

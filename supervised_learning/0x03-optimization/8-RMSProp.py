#!/usr/bin/env python3
"""Contains the create_RMSProp_op function"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: small number to avoid division by zero
    :return: RMSProp optimization operation
    """
    return (tf.train.RMSPropOptimizer(learning_rate=alpha,
                                      decay=beta2,
                                      epsilon=epsilon).minimize(loss))

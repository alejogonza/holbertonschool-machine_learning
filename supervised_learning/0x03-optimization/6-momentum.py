#!/usr/bin/env python3
"""Contains the create_momentum_op function"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm:
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: momentum weight
    :return: the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)

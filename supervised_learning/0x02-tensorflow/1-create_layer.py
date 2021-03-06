#!/usr/bin/env python3
"""Contains the create_layer function"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    create layer function
    :param prev: tensor output of the previous layer
    :param n: number of nodes in the layer to create
    :param activation: activation function that the layer should use
    :return: tensor output of the layer
    """
    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')

    return model(prev)

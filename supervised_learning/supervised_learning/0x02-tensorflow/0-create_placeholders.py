#!/usr/bin/env python3
"""Contains the create_placeholders function"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y, for the neural network
    :param nx: number of feature columns in our data
    :param classes: number of classes in our classifier
    :return: placeholders named x and y, respectively
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y

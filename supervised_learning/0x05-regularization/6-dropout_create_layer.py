#!/usr/bin/env python3
"""Contains the dropout_create_layer"""

import numpy as np
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation function that should be used on the layer
    :param keep_prob: probability that a node will be kept
    :return: output of the new layer
    """
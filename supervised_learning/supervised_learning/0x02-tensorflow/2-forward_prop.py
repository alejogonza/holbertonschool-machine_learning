#!/usr/bin/env python3
"""Contains the forward_prop function"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    :param x: placeholder for the input data
    :param layer_sizes: list containing the number of nodes in
        each layer of the network
    :param activations: list containing the activation functions
        for each layer of the network
    :return: prediction of the network in tensor form
    """

    # first layer activation with features x as input
    y_pred = create_layer(x, layer_sizes[0], activations[0])

    # successive layers activations with y_pred from the prev layer as input
    for i in range(1, len(layer_sizes)):
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])

    return y_pred

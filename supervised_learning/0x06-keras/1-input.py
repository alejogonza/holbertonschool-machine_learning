#!/usr/bin/env python3
"""Contains the build_model function"""

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes
        in each layer of the network
    :param activations: list containing the activation
        functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """
    # input placeholder
    inputs = keras.Input(shape=(nx,))

    # regularization scheme
    reg = keras.regularizers.L1L2(l2=lambtha)

    # a layer instance is callable on a tensor, and returns a tensor.
    # first densely-connected layer
    my_layer = keras.layers.Dense(units=layers[0],
                                  activation=activations[0],
                                  kernel_regularizer=reg,
                                  input_shape=(nx,))(inputs)

    # subsequent densely-connected layers:
    for i in range(1, len(layers)):
        my_layer = keras.layers.Dropout(1 - keep_prob)(my_layer)
        my_layer = keras.layers.Dense(units=layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=reg,
                                      )(my_layer)

    network = keras.Model(inputs=inputs, outputs=my_layer)

    return network

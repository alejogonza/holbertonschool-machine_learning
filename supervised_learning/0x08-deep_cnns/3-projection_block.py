#!/usr/bin/env python3
"""contains the projection_block function"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    builds a projection block
    :param A_prev:
    :param filters: tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    :param s: stride of the first convolution in both the main path
        and the shortcut connection
    :return: activated output of the identity block
    """
    F11, F3, F12 = filters

    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer,
                               )(A_prev)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 3x3
    my_layer = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)

    # shotcut path
    # Conv 1x1
    short = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding='same',
                            kernel_initializer=initializer,
                            )(A_prev)

    short = K.layers.BatchNormalization(axis=3)(short)

    output = K.layers.Add()([my_layer, short])

    output = K.layers.Activation('relu')(output)

    return output

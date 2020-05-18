#!/usr/bin/env python3
"""Contains the calculate_loss function"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction
    :param y: placeholder for the labels of the input data
    :param y_pred: tensor containing the network’s predictions
    :return: tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)

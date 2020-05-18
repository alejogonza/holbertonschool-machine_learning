#!/usr/bin/env python3
"""Contains the calculate_accuracy function"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction:
    :param y: placeholder for the labels of the input data
    :param y_pred: tensor containing the network’s predictions
    :return: tensor containing the decimal accuracy of the prediction
    """
    # from one y_pred one_hot to tag
    y_pred_t = tf.argmax(y_pred, 1)

    # from y one_hot to tag
    y_t = tf.argmax(y, 1)

    # comparison vector between tags (TRUE/FALSE)
    equal = tf.equal(y_pred_t, y_t)

    # average hits
    mean = tf.reduce_mean(tf.cast(equal, tf.float32))
    return mean

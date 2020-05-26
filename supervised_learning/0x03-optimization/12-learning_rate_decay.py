#!/usr/bin/env python3
"""Contains the learning_rate_decay"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow using
        inverse time decay:
    :param alpha: the original learning rate
    :param decay_rate: weight used to determine the rate at
        which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: learning rate decay operation
    """
    alpha = tf.train.inverse_time_decay(alpha,
                                        global_step,
                                        decay_step,
                                        decay_rate,
                                        staircase=True)

    return alpha

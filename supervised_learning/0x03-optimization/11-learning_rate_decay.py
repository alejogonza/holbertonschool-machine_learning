#!/usr/bin/env python3
"""Contains the learning_rate_decay function"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in numpy
    :param alpha: original learning rate
    :param decay_rate: weight used to determine
        the rate at which alpha will decay
    :param global_step: number of passes of gradient descent
        that have elapsed
    :param decay_step: number of passes of gradient descent that
        should occur before alpha is decayed further
    :return: updated value for alpha
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))

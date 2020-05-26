#!/usr/bin/env python3
"""Contains the update_variables_RMSProp function"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: small number to avoid division by zero
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: previous first moment of var
    :param s: previous second moment of var
    :param t: time step used for bias correction
    :return: updated variable, the new first moment,
        and the new second moment, respectively
    """
    # Exponentially Weighted Averages (momentum)
    v = beta1 * v + (1 - beta1) * grad

    # RMSProp
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # bias correction
    v_corrected = v / (1 - (beta1 ** t))
    s_corrected = s / (1 - (beta2 ** t))

    # variable update with ADAM (adaptive moment estimation)
    var = var - alpha * v_corrected / (s_corrected ** (1/2) + epsilon)

    return var, v, s

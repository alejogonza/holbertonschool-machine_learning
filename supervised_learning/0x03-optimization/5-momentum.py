#!/usr/bin/env python3
"""Contains the update_variables_momentum function"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with
    momentum optimization algorithm
    :param alpha: learning rate
    :param beta1: momentum weight
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: previous first moment of var
    :return: updated variable and the new moment, respectively
    """
    # Exponentially Weighted Averages
    v = beta1 * v + (1 - beta1) * grad

    # variable update
    var = var - alpha * v
    return var, v

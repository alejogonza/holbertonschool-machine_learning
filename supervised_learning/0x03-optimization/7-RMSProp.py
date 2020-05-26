#!/usr/bin/env python3
"""Contains the update_variables_RMSProp function"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon:  small number to avoid division by zero
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param s: previous second moment of var
    :return: updated variable and the new moment, respectively
    """
    # RMSProp
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # variable update
    var = var - alpha * grad / (s ** (1/2) + epsilon)

    return var, s

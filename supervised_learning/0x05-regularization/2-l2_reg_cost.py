#!/usr/bin/env python3
"""Contains the l2_reg_cost function"""

import numpy as np

def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:
    :param cost: tensor containing the cost of the network without L2 regularization
    :return: tensor containing the cost of the network accounting for L2 regularization
    """
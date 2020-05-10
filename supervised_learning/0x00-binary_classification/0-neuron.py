#!/usr/bin/env python3
"""Contains the Neuron class"""

import numpy as np


class Neuron:
    """Neuron class
    nx is the number of input features to the neuron
    """

    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0

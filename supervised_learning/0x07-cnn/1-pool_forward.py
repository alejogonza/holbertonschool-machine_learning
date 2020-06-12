#!/usr/bin/env python3
"""Contains the pool_forward function"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a convolutional layer of a neural network
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param kernel_shape: is a tuple of (kh, kw)
        containing the size of the kernel for the pooling
        containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
    :param stride: tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    _:param mode: string containing either max or avg,
        indicating whether to perform maximum or average pooling, respectively
    :return: output of the convolutional layer
    """
    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from kerner_shape
    (kh, kw) = kernel_shape

    # Retrieving stride
    (sh, sw) = stride

    # Compute the output dimensions
    n_h = int(((h_prev - kh) / sh) + 1)
    n_w = int(((w_prev - kw) / sw) + 1)

    # Initialize the output with zeros
    output = np.zeros((m, n_h, n_w, c_prev))

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for x in range(n_w):
        for y in range(n_h):
            # element-wise multiplication of the kernel and the image
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(A_prev[:,
                           y * sh: y * sh + kh,
                           x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(A_prev[:,
                            y * sh: y * sh + kh,
                            x * sw: x * sw + kw], axis=(1, 2))

    return output

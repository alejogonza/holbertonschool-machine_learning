#!/usr/bin/env python3
"""
Determinant
"""


def minor_m(m, row, col):
    """Omite the the given row and column of a square matrix.

    Args:
        m (list): matrix.
        row (int): row to omite.
        col (int): column to omite.

    Returns:
        the matrix with the omited row, column.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]


def determinant(matrix):
    """ Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        the determinant.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix[0])):
        omited_matrix = minor_m(matrix, 0, j)
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det


def transponse(m):
    """ Transpose a matrix

    Args:
        m (list): matrix.

    Returns:
        the transposed matrix
    """
    return [[m[row][col] for row in range(len(m))]
            for col in range(len(m[0]))]


def adjugate(matrix):
    """ Calculates the adjugate of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        the adjugate.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if (matrix[0] and len(matrix) != len(matrix[0])) or \
            matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    cofactor = [[((-1) ** (i + j)) * determinant(minor_m(matrix, i, j))
                 for j in range(len(matrix[i]))] for i in range(len(matrix))]

    return transponse(cofactor)


def inverse(matrix):
    """Calculate inverse matrix.

    Args:
        matrix (list): matrix.

    Returns:
        inverse matrix.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if (len(matrix) == 0 or len(matrix) != len(matrix[0])) \
            or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if any([len(l) != len(matrix) for l in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adjugated = adjugate(matrix)

    return [[n / det for n in row] for row in adjugated]

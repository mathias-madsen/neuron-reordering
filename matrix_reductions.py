"""
This module contains functions for constructing matrices that represent
the operation of taking row sums or column sums of other matrices.
"""

import numpy as np


def get_row_sum_matrix(height, width):
    """ Get a matrix M such that M @ B.flatten() == sum(B, axis=1).
    
    Returns a matrix of shape `[height, height * width]`.
    """
    return np.concatenate([np.tile(e, [width, 1])
                           for e in np.eye(height)], axis=0).T


def get_col_sum_matrix(height, width):
    """ Get the matrix M such that M @ B.flatten() == sum(B, axis=0).
    
    Returns a matrix of shape `[width, height * width]`.
    """
    return np.tile(np.eye(width), [height, 1]).T


def get_row_col_sum_matrix(height, width):
    """ Get matrix row- and col-sums extractor matrix.
    
    Returns a matrix of shape `[height + width, height * width]`.
    """

    row_sum_mat = get_row_sum_matrix(height, width)
    col_sum_mat = get_col_sum_matrix(height, width)

    return np.concatenate([row_sum_mat, col_sum_mat], axis=0)


def _test_row_sum_matrix():

    for _ in range(10):
        height, width = np.random.randint(1, 10, size=2)
        matrix = np.random.normal(size=(height, width))
        extractor = get_row_sum_matrix(height, width)
        assert extractor.shape == (height, height * width)
        rowsums = extractor @ matrix.flatten()
        assert np.allclose(rowsums, matrix.sum(axis=1))


def _test_col_sum_matrix():

    for _ in range(10):
        height, width = np.random.randint(1, 10, size=2)
        matrix = np.random.normal(size=(height, width))
        extractor = get_col_sum_matrix(height, width)
        assert extractor.shape == (width, height * width)
        colsums = extractor @ matrix.flatten()
        assert np.allclose(colsums, matrix.sum(axis=0))


def _test_sum_matrix():

    for _ in range(10):
        height, width = np.random.randint(1, 10, size=2)
        matrix = np.random.normal(size=(height, width))
        extractor = get_row_col_sum_matrix(height, width)
        assert extractor.shape == (height + width, height * width)
        sums = extractor @ matrix.flatten()
        assert np.allclose(sums[:height], matrix.sum(axis=1))  # rowsums
        assert np.allclose(sums[height:], matrix.sum(axis=0))  # colsums

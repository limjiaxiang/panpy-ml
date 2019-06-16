import sys

import numpy as np


def scale_matrix(approach, matrix, prior_scale_arrays=None, return_scale_arrays=False):
    if approach == 'mean':
        output_matrix, scale_arrays = mean_normalise(matrix, prior_minmax=prior_scale_arrays, return_minmax=True)
    elif approach == 'minmax':
        output_matrix, scale_arrays = minmax_scale(matrix, prior_mean_std=prior_scale_arrays, return_mean_std=True)
    else:
        sys.exit('Scaling approach not found')
    if return_scale_arrays:
        return output_matrix, scale_arrays, approach
    return output_matrix


def mean_normalise(matrix, prior_minmax=None, return_minmax=True):
    if prior_minmax:
        min_array = prior_minmax[0]
        max_array = prior_minmax[1]
    else:
        min_array = np.min(matrix, axis=0)
        max_array = np.max(matrix, axis=0)
    if return_minmax:
        return (np.divide(np.subtract(matrix, min_array), np.subtract(max_array, min_array)),
                (np.min(matrix, axis=0), np.max(matrix, axis=0)))
    return np.divide(np.subtract(matrix, min_array), np.subtract(max_array, min_array))


def minmax_scale(matrix, prior_mean_std=None, return_mean_std=True):
    if prior_mean_std:
        mean_array = prior_mean_std[0]
        std_array = prior_mean_std[1]
    else:
        mean_array = np.mean(matrix, axis=0)
        std_array = np.std(matrix, axis=0)
    if return_mean_std:
        return (np.divide(np.subtract(matrix, mean_array), std_array),
                (mean_array, std_array))
    return np.divide(np.subtract(matrix, mean_array), std_array)

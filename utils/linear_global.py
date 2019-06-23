import numpy as np

from utils.scaling import scale_matrix


def linear_predict(x_matrix, params, scale_approach=None, prior_scale_arrays=None):
    # multivariate linear regression
    x_matrix_copy = np.copy(x_matrix)
    if scale_approach:
        x_matrix_copy = scale_matrix(scale_approach, x_matrix_copy,
                                     prior_scale_arrays=prior_scale_arrays, return_scale_arrays=False)
    x_vector_with_bias = check_bias_column(x_matrix_copy)
    return np.dot(x_vector_with_bias, params)


def check_bias_column(matrix):
    if np.all(matrix[:, 0] != 1.0, axis=0):
        vector_copy = np.copy(matrix)
        vector_with_bias = np.insert(vector_copy, obj=0, values=1.0, axis=1)
        print('Added bias column 1.0 at column index 0')
        return vector_with_bias
    else:
        return matrix

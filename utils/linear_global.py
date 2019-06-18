import numpy as np

from utils.scaling import scale_matrix


def linear_predict(x_vector, params, scale_approach=None, prior_scale_arrays=None):
    # multivariate linear regression
    x_vector_copy = np.copy(x_vector)
    if scale_approach:
        x_vector_copy = scale_matrix(scale_approach, x_vector_copy,
                                     prior_scale_arrays=prior_scale_arrays, return_scale_arrays=False)
    x_vector_with_bias = np.insert(x_vector_copy, obj=0, values=1.0, axis=1)
    return np.dot(x_vector_with_bias, params)

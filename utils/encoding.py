import numpy as np


def one_hot(vector):
    num_rows = vector.shape[0]
    num_unique = np.unique(vector).shape[0]
    one_hot_matrix = np.zeros((num_rows, num_unique))
    one_hot_matrix[np.arange(num_rows), vector] = 1
    return one_hot_matrix

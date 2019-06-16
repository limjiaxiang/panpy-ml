import sys

import numpy as np


class GradientDescent:

    def __init__(self, cost_function, lr=0.01, epochs=100, batch_size=32, random_seed=None):
        self.cost_function = cost_function
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = None
        if random_seed is None:
            random_seed = np.random.randint(2**32-1)
        np.random.seed(random_seed)

    def _calculate_cost(self, params, x_matrix, y_matrix, scale_x=True):
        return self.cost_function(params, x_matrix, y_matrix, scale_x=scale_x)

    # initialise parameters to be [0, 1)
    def _init_params(self, size):
        self.params = np.random.rand(size, 1).flatten()

    def descent(self, x_matrix, y_matrix):
        if self.params is None:
            self._init_params(x_matrix.shape[1])
        total_examples = x_matrix.shape[0]
        curr_epoch = 0
        curr_batch_index = 0
        while curr_epoch < self.epochs:
            if curr_batch_index + self.batch_size >= total_examples:
                x_batch = x_matrix[curr_batch_index:]
                y_batch = y_matrix[curr_batch_index:]
                curr_batch_index = 0
                curr_epoch += 1
            else:
                x_batch = x_matrix[curr_batch_index: curr_batch_index + self.batch_size]
                y_batch = y_matrix[curr_batch_index: curr_batch_index + self.batch_size]
                curr_batch_index += self.batch_size
            batch_pred = np.dot(x_batch, self.params)
            batch_error = np.subtract(batch_pred, y_batch)
            update_vector = np.multiply((self.learning_rate / self.batch_size),
                                        np.sum(np.dot(x_batch.T, batch_error)))
            self.params -= update_vector


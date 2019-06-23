import sys

import numpy as np


class GradientDescent:

    def __init__(self, model, lr=0.01, max_epochs=10000, batch_size=32, random_seed=None):
        self.model = model
        self.learning_rate = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        if random_seed is None:
            random_seed = np.random.randint(2**32-1)
        np.random.seed(random_seed)

    def _calculate_cost(self, params, x_matrix, y_matrix, scale_x=False):
        return self.model.cost_function(params, x_matrix, y_matrix, scale_x=scale_x)

    # initialise parameters to be [-1, 1)
    def _init_params(self, size):
        self.model.params = (np.random.rand(size, 1).flatten() * 2.0) - 1.0

    def descent(self, x_matrix, y_matrix, print_epoch_info=True):
        if self.model.params is None:
            self._init_params(x_matrix.shape[1])
        total_examples = x_matrix.shape[0]
        curr_epoch = 0
        curr_batch_index = 0
        curr_cost = sys.maxsize
        new_epoch = True
        while curr_epoch < self.max_epochs:
            if curr_batch_index + self.batch_size >= total_examples:
                x_batch = x_matrix[curr_batch_index:]
                y_batch = y_matrix[curr_batch_index:]
                curr_batch_index = 0
                curr_epoch += 1
                new_epoch = True
            else:
                x_batch = x_matrix[curr_batch_index: curr_batch_index + self.batch_size]
                y_batch = y_matrix[curr_batch_index: curr_batch_index + self.batch_size]
                curr_batch_index += self.batch_size
            batch_pred = np.dot(x_batch, self.model.params)
            batch_error = np.subtract(batch_pred, y_batch)
            update_vector = np.multiply((self.learning_rate / self.batch_size),
                                        np.dot(x_batch.T, batch_error))

            # check if cost function is converging
            new_params = self.model.params - update_vector
            new_params_cost = self._calculate_cost(new_params, x_matrix=x_matrix, y_matrix=y_matrix, scale_x=False)
            if new_params_cost > curr_cost:
                print('Gradient descent ceased to converge on cost function')
                break
            elif new_params_cost <= 1e-3:
                print(f'Cost has grown exceedingly small: 0.001 >= {new_params_cost}')
                break
            else:
                self.model.params = new_params
                curr_cost = new_params_cost
                if new_epoch and print_epoch_info:
                    print(f'Start of Epoch {curr_epoch} Cost: {curr_cost:.2f}, Parameters: {self.model.params}')
                    new_epoch = False

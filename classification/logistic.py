import numpy as np
import pandas as pd

from utils.linear_global import linear_predict
from utils.scaling import scale_matrix
from utils.formulae import sigmoid
from regression.optimisation import GradientDescent


class LogisticRegression:

    def __init__(self, random_seed=None):
        self.train_x = None
        self.train_y = None
        self.params = None
        self.scale_arrays = None
        self.scale_approach = None
        self.random_seed = random_seed

    def fit(self, train_x, train_y, scale_approach=None, gradient_args=None):
        # TODO implement fit
        gr = GradientDescent(self._cost_function, random_seed=self.random_seed, **gradient_args)
        pass

    def predict(self, x_vector):
        logit = linear_predict(x_vector, self.params, scale_approach=self.scale_approach,
                               prior_scale_arrays=self.scale_arrays)
        return sigmoid(logit)

    def _cost_function(self, x_matrix, y_matrix, scale_x=True):
        if scale_x:
            x_matrix = scale_matrix(self.scale_approach, x_matrix,
                                    prior_scale_arrays=self.scale_arrays, return_scale_arrays=False)
        hypo = self.predict(x_matrix)
        y_comp = np.multiply(y_matrix, np.log(hypo))
        y_minus_comp = np.multiply(np.subtract(1, y_matrix), np.log(np.subtract(1, hypo)))
        total_cost = (-1/y_matrix.shape[0]) * (y_comp + y_minus_comp)
        return total_cost


if __name__ == '__main__':
    # TODO validate log reg model
    pass

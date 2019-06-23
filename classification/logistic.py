import numpy as np
import pandas as pd

from utils.linear_global import linear_predict, check_bias_column
from utils.scaling import scale_matrix
from utils.formulae import sigmoid
from regression.optimisation import GradientDescent


class LogisticRegression:

    # multi-class compatible logistic regression (ovr and cross-entropy loss [multinomial])
    def __init__(self, Lasso=False, Ridge=False, l1_ratio=None, random_seed=None):
        self.train_x = None
        self.train_y = None
        self.params = None
        self.scale_arrays = None
        self.scale_approach = None
        self.random_seed = random_seed

    def fit(self, train_x, train_y, scale_approach=None, gradient_args=None, decision_boundary=0.5):
        # TODO implement fit
        self.train_x = train_x.values if isinstance(train_x, pd.DataFrame) else train_x
        self.train_y = train_y.values if isinstance(train_y, pd.DataFrame) else train_y
        self.scale_approach = scale_approach
        if scale_approach:
            temp_x, self.scale_arrays, self.scale_approach = \
                scale_matrix(scale_approach, self.train_x, prior_scale_arrays=False, return_scale_arrays=True)
        else:
            temp_x = self.train_x
        temp_x = check_bias_column(temp_x)
        if gradient_args:
            gr = GradientDescent(self, random_seed=self.random_seed, **gradient_args)
        else:
            gr = GradientDescent(self, random_seed=self.random_seed)
        gr.descent(x_matrix=temp_x, y_matrix=self.train_y)

    def predict(self, x_matrix, scale=True, custom_params=None, binary=False):
        predict_params = custom_params if custom_params is not None else self.params
        if scale:
            logit = linear_predict(x_matrix, predict_params, scale_approach=self.scale_approach,
                                   prior_scale_arrays=self.scale_arrays)
        else:
            logit = linear_predict(x_matrix, predict_params)
        output = sigmoid(logit)
        if binary:
            output = np.where(output >= 0.5, 1, 0)
        return output

    def cost_function(self, params, x_matrix, y_matrix, scale_x=True):
        if scale_x:
            x_matrix = scale_matrix(self.scale_approach, x_matrix,
                                    prior_scale_arrays=self.scale_arrays, return_scale_arrays=False)
        hypo = self.predict(x_matrix, scale=False, custom_params=params)
        y_comp = np.multiply(y_matrix, np.log(hypo))
        y_minus_comp = np.multiply(np.subtract(1, y_matrix), np.log(np.subtract(1, hypo)))
        sum_comps = y_comp + y_minus_comp
        cost = (-1/y_matrix.shape[0]) * np.sum(sum_comps)
        return cost


if __name__ == '__main__':
    RANDOM_SEED = 15
    # TODO validate log reg model
    # Iris dataset
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # this module's log reg class
    lr = LogisticRegression(random_seed=RANDOM_SEED)
    lr.fit(X_train, y_train, scale_approach='mean')
    lr_pred = lr.predict(X_test, binary=True)

    # sklearn log reg class
    sk_lr = linear_model.LogisticRegression(penalty='none', random_state=RANDOM_SEED, solver='saga',
                                            max_iter=100, multi_class='ovr')
    ss = StandardScaler()
    sk_X_train = ss.fit_transform(X_train)
    sk_lr.fit(sk_X_train, y_train)
    sk_lr_pred = sk_lr.predict(ss.transform(X_test))

    print('hello')

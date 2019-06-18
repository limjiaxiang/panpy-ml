import numpy as np
import pandas as pd

from utils.linear_global import linear_predict
from utils.scaling import scale_matrix
from utils.formulae import pearson_correlation
from regression.optimisation import GradientDescent


# Multivariate Linear Regression
class LinearRegression:

    def __init__(self, Lasso=False, Ridge=False, random_seed=None):
        self.train_x = None
        self.train_y = None
        self.params = None
        self.scale_arrays = None
        self.scale_approach = None
        self.random_seed = random_seed

    # multivariate linear regression
    # obtain parameters for model (thetas)
    def fit(self, train_x, train_y, scale_approach=None, method='gradient', gradient_args=None):
        # TODO ensure insertion logic passes, solution: create 1st column checker for bias placeholder value
        if gradient_args is None:
            gradient_args = dict(lr=0.01, epochs=1000, batch_size=32)
        self.train_x = train_x.values
        self.train_y = train_y.values
        self.scale_approach = scale_approach
        if scale_approach:
            temp_x, self.scale_arrays, self.scale_approach = \
                scale_matrix(scale_approach, self.train_x, prior_scale_arrays=False, return_scale_arrays=True)
            temp_x = np.insert(temp_x, obj=0, values=1.0, axis=1)
        else:
            temp_x = np.insert(self.train_x, obj=0, values=1.0, axis=1)
        if method == 'normal':
            self._normal_eqn(temp_x, self.train_y)
        # https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
        elif method == 'gradient':
            gr = GradientDescent(self._cost_function, random_seed=self.random_seed, **gradient_args)
            gr.descent(x_matrix=temp_x, y_matrix=self.train_y)
            self.params = gr.params

    def predict(self, x_vector):
        return linear_predict(x_vector, self.params, scale_approach=self.scale_approach,
                              prior_scale_arrays=self.scale_arrays)

    # obtain linear regression parameters through normal equations (closed-form), a.k.a. OLS
    # https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/
    # theta (params) = (X^2)^-1 * (X*y)
    def _normal_eqn(self, x, y):
        x_squared_inverse = np.linalg.inv(np.dot(x.T, x))
        x_y_product = np.dot(x.T, y)
        self.params = np.dot(x_squared_inverse, x_y_product)
        print('Parameters obtained obtained from normal equation')

    def _cost_function(self, params, x_matrix, y_matrix, scale_x=True):
        if scale_x:
            x_matrix = scale_matrix(self.scale_approach, x_matrix,
                                    prior_scale_arrays=self.scale_arrays, return_scale_arrays=False)
        scaled_x_matrix_w_bias = np.insert(x_matrix, obj=0, values=1.0, axis=1)
        return (1/(2*scaled_x_matrix_w_bias.shape[0])) * \
               np.sum(np.square(np.subtract(np.dot(scaled_x_matrix_w_bias, params), y_matrix)), axis=0)


class SimpleLinear:

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.intercept = None
        self.param = None

    def simple_fit(self, train_x, train_y):
        """
        Simple linear regression (1 feature)
        y = a + bx
        b = r * (sd_y/sd_x)
        where:
          r is Pearson's correlation coefficient
          sd_y is standard deviation of y
          sd_x is standard deviation of x
        a (y-intercept) = y_mean - b*x_mean
        :param train_x: vector of predictor, independent feature
        :param train_y: vector of outcome, dependent feature
        """
        pearson, train_x_sd, train_y_sd = pearson_correlation(train_x, train_y, generate_sd=True)
        # regression slope: b
        self.param = np.multiply(pearson, np.divide(train_y_sd, train_x_sd))
        # y-intercept: a
        self.intercept = np.mean(y, axis=0) - np.multiply(self.param, np.mean(x, axis=0))

    def simple_predict(self, x_vector):
        """
        Returns prediction vector
        y_hat = a + bx
        :param x_vector: vector of predictor variable's values
        :return: vector of predicted values
        """
        predict_vector = self.intercept + np.multiply(self.param, x_vector)
        return predict_vector


if __name__ == '__main__':
    # Head Dimensions in Brothers (Multivariate)
    # frets = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/boot/frets.csv')
    # x = frets.iloc[:, 1:-1]
    # y = frets.iloc[:, -1]
    # lr = LinearRegression()
    # lr.fit(x, y, scale_approach='mean', method='gradient', gradient_args={'epochs': 1000})
    # pred = lr.predict(x.values)

    # # Example Data of Antille and May - for Simple Regression
    # exAM = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/exAM.csv')
    # x = exAM.iloc[:, 1]
    # y = exAM.iloc[:, -1]
    # lr = LinearRegression()
    # lr.simple_fit(x, y)
    # prediction = lr.simple_predict(x)

    # Boston house dataset
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.datasets import load_boston

    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    features = ['LSTAT', 'RM']
    target = boston['MEDV']

    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
    Y = boston['MEDV']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

    # current module lr class
    lr = LinearRegression()
    lr.fit(X_train, Y_train, scale_approach='l2_norm', method='normal')
    lr_pred = lr.predict(X_test)
    # sklearn lr class
    sk_lr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    sk_lr.fit(X_train, Y_train)
    sk_pred = sk_lr.predict(X_test)

    print('hellO!')

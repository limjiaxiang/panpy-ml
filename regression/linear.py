import numpy as np
import pandas as pd

from utils.linear_global import linear_predict, check_bias_column
from utils.scaling import scale_matrix
from utils.formulae import pearson_correlation
from regression.optimisation import GradientDescent


# Multivariate Linear Regression
class LinearRegression:

    def __init__(self, lasso_lambda=0.0, ridge_lambda=0.0, l1_ratio=0.5, random_seed=None):
        self.train_x = None
        self.train_y = None
        self.params = None
        self.scale_arrays = None
        self.scale_approach = None
        self.random_seed = random_seed
        self.lambdas = {
            'lasso': lasso_lambda,
            'ridge': ridge_lambda
        }
        self.l1_ratio = l1_ratio

    # multivariate linear regression
    # obtain parameters for model (thetas)
    def fit(self, train_x, train_y, scale_approach=None, method='gradient', gradient_args=None):
        self.train_x = train_x.values if isinstance(train_x, pd.DataFrame) else train_x
        self.train_y = train_y.values if isinstance(train_y, pd.DataFrame) else train_y
        self.scale_approach = scale_approach
        if scale_approach:
            temp_x, self.scale_arrays, self.scale_approach = \
                scale_matrix(scale_approach, self.train_x, prior_scale_arrays=False, return_scale_arrays=True)
        else:
            temp_x = self.train_x
        temp_x = check_bias_column(temp_x)
        if method == 'normal':
            self._normal_eqn(temp_x, self.train_y)
        # https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
        elif method == 'gradient':
            if gradient_args:
                gr = GradientDescent(self, random_seed=self.random_seed, **gradient_args)
            else:
                gr = GradientDescent(self, random_seed=self.random_seed)
            self.params = gr.descent(x_matrix=temp_x, y_matrix=self.train_y)

    def predict(self, x_matrix, scale=True, custom_params=None, **kwargs):
        predict_params = custom_params if custom_params is not None else self.params
        if scale:
            pred = linear_predict(x_matrix, predict_params, scale_approach=self.scale_approach,
                                  prior_scale_arrays=self.scale_arrays)
        else:
            pred = linear_predict(x_matrix, predict_params)
        return pred

    # obtain linear regression parameters through normal equations (closed-form), a.k.a. OLS
    # https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/
    # theta (params) = (X^2)^-1 * (X*y)
    def _normal_eqn(self, x, y):
        x_squared_inverse = np.linalg.inv(np.dot(x.T, x))
        x_y_product = np.dot(x.T, y)
        self.params = np.dot(x_squared_inverse, x_y_product)
        print('Parameters obtained obtained from normal equation')

    def cost_function(self, params, x_matrix, y_matrix, scale_x=True, **kwargs):
        if scale_x:
            x_matrix = scale_matrix(self.scale_approach, x_matrix,
                                    prior_scale_arrays=self.scale_arrays, return_scale_arrays=False)
        scaled_x_matrix_w_bias = check_bias_column(x_matrix)
        cost = (1/(2*scaled_x_matrix_w_bias.shape[0])) * \
               np.sum(np.square(np.subtract(np.dot(scaled_x_matrix_w_bias, params), y_matrix)), axis=0)
        params_wo_intercept = params[1:]
        lasso_cost = self.lambdas['lasso'] * np.sum(np.abs(params_wo_intercept))
        ridge_cost = self.lambdas['ridge'] * np.sum(np.square(params_wo_intercept))
        if self.lambdas['lasso'] > 0.0 and self.lambdas['ridge'] > 0.0:
            lasso_cost *= self.l1_ratio
            ridge_cost *= (1 - self.l1_ratio)
        cost += lasso_cost + ridge_cost
        return cost


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
    RANDOM_SEED = 15

    # Head Dimensions in Brothers (Multivariate)
    # frets = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/boot/frets.csv')
    # x = frets.iloc[:, 1:-1]
    # y = frets.iloc[:, -1]
    # lr = LinearRegression()
    # lr.fit(x, y, scale_approach='mean', method='gradient')
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
    y = boston['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # current module lr class
    lr = LinearRegression(random_seed=RANDOM_SEED)
    lr.fit(X_train, y_train, scale_approach='l2_norm', method='gradient')
    lr_pred = lr.predict(X_test)

    # current module lr class with lasso reg
    lr_lasso = LinearRegression(lasso_lambda=10, random_seed=RANDOM_SEED)
    lr_lasso.fit(X_train, y_train, scale_approach='l2_norm', method='gradient')
    lr_lasso_pred = lr_lasso.predict(X_test)

    # current module lr class with ridge reg
    lr_ridge = LinearRegression(ridge_lambda=10, random_seed=RANDOM_SEED)
    lr_ridge.fit(X_train, y_train, scale_approach='l2_norm', method='gradient')
    lr_ridge_pred = lr_ridge.predict(X_test)

    # current module lr class with elastic reg
    lr_elastic = LinearRegression(lasso_lambda=10, ridge_lambda=10, l1_ratio=0.5, random_seed=RANDOM_SEED)
    lr_elastic.fit(X_train, y_train, scale_approach='l2_norm', method='gradient')
    lr_elastic_pred = lr_elastic.predict(X_test)

    # sklearn lr class
    sk_lr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    sk_lr.fit(X_train, y_train)
    sk_pred = sk_lr.predict(X_test)

    print('hellO!')

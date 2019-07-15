import numpy as np
import pandas as pd

from utils.linear_global import linear_predict, check_bias_column
from utils.scaling import scale_matrix
from utils.formulae import sigmoid
from regression.optimisation import GradientDescent


class LogisticRegression:

    # multi-class compatible logistic regression (ovr)
    def __init__(self, Lasso=False, Ridge=False, l1_ratio=None, random_seed=None, decision_boundary=0.5,
                 multi_type='ovr'):
        self.train_x = None
        self.train_y = None
        self.scale_arrays = None
        self.scale_approach = None
        self.decision_boundary = decision_boundary
        self.random_seed = random_seed
        self.params = None
        self.multi_type = multi_type
        self.is_multi = False

    def fit(self, train_x, train_y, scale_approach=None, gradient_args=None):
        self.train_x = train_x.values if isinstance(train_x, pd.DataFrame) else train_x
        self.train_y = train_y.values if isinstance(train_y, pd.DataFrame) else train_y
        self.scale_approach = scale_approach
        if scale_approach:
            temp_x, self.scale_arrays, self.scale_approach = \
                scale_matrix(scale_approach, self.train_x, prior_scale_arrays=False, return_scale_arrays=True)
        else:
            temp_x = self.train_x
        temp_x = check_bias_column(temp_x)
        if np.unique(self.train_y).shape[0] > 2:
            self.is_multi = True
            self.params = {y_label: None for y_label in np.unique(self.train_y)}
            if self.multi_type == 'ovr':
                self.ovr(temp_x, gradient_args)
        else:
            gr = (GradientDescent(self, random_seed=self.random_seed, **gradient_args) if gradient_args
                  else GradientDescent(self, random_seed=self.random_seed))
            self.params = gr.descent(x_matrix=temp_x, y_matrix=self.train_y)

    def predict(self, x_matrix, scale=True, custom_params=None, binary=False, train=False):
        output = None
        if self.is_multi and not train:
            if self.multi_type == 'ovr':
                index_label_mapping = {index: y_label for index, y_label in enumerate(np.unique(self.train_y))}
                pred_matrix = np.empty((x_matrix.shape[0], len(index_label_mapping)))
                for index, y_label in index_label_mapping.items():
                    y_label_params = self.params[y_label]
                    y_label_output = self._logit_predict(x_matrix, y_label_params, scale,
                                                         scale_approach=self.scale_approach,
                                                         prior_scale_arrays=self.scale_arrays)
                    pred_matrix[:, index] = y_label_output
                pred_vector_index = np.argmax(pred_matrix, axis=1)
                pred_vector_labels = np.array(list(map(lambda index: index_label_mapping[index], pred_vector_index)))
                output = pred_vector_labels
            else:
                print('multi_type not specified')
        else:
            predict_params = custom_params if custom_params is not None else self.params
            output = self._logit_predict(x_matrix, predict_params, scale,
                                         scale_approach=self.scale_approach, prior_scale_arrays=self.scale_arrays)
            if binary:
                output = np.where(output >= self.decision_boundary, 1, 0)
        return output

    def _logit_predict(self, x_matrix, predict_params, scale, scale_approach, prior_scale_arrays):
        if scale:
            logit = linear_predict(x_matrix, predict_params,
                                   scale_approach=scale_approach, prior_scale_arrays=prior_scale_arrays)
        else:
            logit = linear_predict(x_matrix, predict_params)
        output = sigmoid(logit)
        return output

    def cost_function(self, params, x_matrix, y_matrix, scale_x=True, train=True):
        if scale_x:
            x_matrix = scale_matrix(self.scale_approach, x_matrix,
                                    prior_scale_arrays=self.scale_arrays, return_scale_arrays=False)
        preds = self.predict(x_matrix, scale=False, custom_params=params, train=train)
        y_comp = np.multiply(y_matrix, np.log(preds))
        y_minus_comp = np.multiply(np.subtract(1, y_matrix), np.log(np.subtract(1, preds)))
        sum_comps = y_comp + y_minus_comp
        cost = (-1 / y_matrix.shape[0]) * np.sum(sum_comps)
        return cost

    def ovr(self, scaled_x_matrix, gradient_args):
        for y_label in self.params.keys():
            curr_train_y = self._ovr_binarise_y_labels(self.train_y, y_label)
            ovr_gr = (GradientDescent(self, random_seed=self.random_seed, **gradient_args) if gradient_args
                      else GradientDescent(self, random_seed=self.random_seed))
            self.params[y_label] = ovr_gr.descent(x_matrix=scaled_x_matrix, y_matrix=curr_train_y)

    @staticmethod
    def _ovr_binarise_y_labels(y_labels, main_label):
        y_label_binary = y_labels.copy()
        y_label_binary[y_label_binary != main_label] = 0
        return y_label_binary


if __name__ == '__main__':
    RANDOM_SEED = 15
    # Iris dataset
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    # X = iris.data[:100, :2]
    # y = iris.target[:100]
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # this module's log reg class
    lr = LogisticRegression(random_seed=RANDOM_SEED)
    lr.fit(X_train, y_train, scale_approach='mean', gradient_args={'max_epochs': 10000})
    lr_pred = lr.predict(X_test, binary=True)

    # sklearn log reg class
    sk_lr = linear_model.LogisticRegression(penalty='none', random_state=RANDOM_SEED, solver='saga',
                                            max_iter=100, multi_class='ovr')
    ss = StandardScaler()
    sk_X_train = ss.fit_transform(X_train)
    sk_lr.fit(sk_X_train, y_train)
    sk_lr_pred = sk_lr.predict(ss.transform(X_test))

    print('hello')

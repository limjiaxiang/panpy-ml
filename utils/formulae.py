import numpy as np


# Pearson's correlation coefficient
# r = summation[(x - x_mean)(y - y_mean)] / sqrt[summation((x - x_mean)^2) * summation((y - y_mean)^2)]
# http://www.datasciencemadesimple.com/wp-content/uploads/2017/07/CORRELATION-COEFFICIENT-FORMULA.png
def pearson_correlation(feature_1, feature_2, generate_sd=False):
    feature_1_mean = np.mean(feature_1, axis=0)
    feature_2_mean = np.mean(feature_2, axis=0)
    feature_1_minus_mean = feature_1 - feature_1_mean
    feature_2_minus_mean = feature_2 - feature_2_mean
    feature_1_minus_mean_squared = np.square(feature_1_minus_mean)
    feature_2_minus_mean_squared = np.square(feature_2_minus_mean)
    features_diff_product = np.multiply(feature_1_minus_mean, feature_2_minus_mean)
    coefficient = (np.sum(features_diff_product, axis=0) /
                   np.sqrt(np.multiply(np.sum(feature_1_minus_mean_squared, axis=0),
                                       np.sum(feature_2_minus_mean_squared, axis=0))))
    if generate_sd:
        feature_1_sd = np.sqrt(np.sum(feature_1_minus_mean_squared)/(np.size(feature_1_minus_mean_squared) - 1))
        feature_2_sd = np.sqrt(np.sum(feature_2_minus_mean_squared)/(np.size(feature_2_minus_mean_squared) - 1))
        return coefficient, feature_1_sd, feature_2_sd
    return coefficient


# outcome/dependent feature residual (error) function
def residual_error(actual_y_matrix, predicted_y_matrix):
    return np.subtract(actual_y_matrix, predicted_y_matrix)


# Sum of squared residuals
def rss(residual_vector):
    return np.sum(np.square(residual_vector), axis=0)


def l2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))


# sigmoid / logistic function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

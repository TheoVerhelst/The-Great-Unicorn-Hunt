from sklearn import metrics
import numpy as np

def root_mean_squared_log_error(regressor, X, y):
    y_pred = regressor.predict(X)
    y_pred[y_pred < 0] = 0 # Some predictions are negative, that screws up the log
    return metrics.mean_squared_error(np.log(y + 1), np.log(y_pred + 1)) ** 0.5

# Here import your model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import resample
from sklearn import preprocessing
import pandas as pd
import numpy as np
from helpers import rmsle_scorer, root_mean_squared_log_error

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

means = np.nanmean(X, axis=0)
nan_locations = np.where(np.isnan(X))
X[nan_locations] = np.take(means, nan_locations[1])

# Normalize X
X = preprocessing.scale(X)

regressor = GaussianProcessRegressor(copy_X_train=False, alpha=0.01778279410038923,
        kernel=kernels.RationalQuadratic(alpha=1, length_scale=1), n_restarts_optimizer=4, normalize_y=False)
"""
regressor = GaussianProcessRegressor(copy_X_train=False)
parameters = {'kernel':(kernels.RationalQuadratic(),  kernels.RBF(), kernels.WhiteKernel()),
    'alpha': np.logspace(-10, 1, 5),
    'n_restarts_optimizer': range(1,5),
    'normalize_y': [True, False]}
clf = GridSearchCV(regressor, parameters, scoring=rmsle_scorer, verbose=10)
X_train, y_train = resample(X, y, n_samples=500)
clf.fit(X_train, y_train)
print("best_estimator_:", clf.best_estimator_)
print("best_score_:", clf.best_score_)
print("best_params_:", clf.best_params_)
print("best_score_:", clf.best_score_)

regressor.set_params(**clf.best_params_)
"""
X_train, y_train = resample(X, y, n_samples=5000)
regressor.fit(X_train, y_train)

print("Training done, testing...")
# Since we can't load the whole dataset, do batch testing
batch_size = 5000
X_test, y_test = resample(X, y, n_samples=100000)
y_pred = np.ndarray((0,))
for i in range(0, X_test.shape[0], batch_size):
    y_pred = np.hstack((y_pred, regressor.predict(X_test[i: i + batch_size])))
print("RMSLE =", root_mean_squared_log_error(y_test, y_pred)) # Last result: 0.469685

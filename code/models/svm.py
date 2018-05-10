# Here import your model
from sklearn.svm import SVR

from sklearn.model_selection import validation_curve
from sklearn.utils import resample
from sklearn import preprocessing
import pandas as pd
import numpy as np
from helpers import root_mean_squared_log_error, rmsle_scorer

print("load")
dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration_in_minutes'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

means = np.nanmean(X, axis=0)
nan_locations = np.where(np.isnan(X))
X[nan_locations] = np.take(means, nan_locations[1])

print("preprocess")
# Normalize X
X = preprocessing.scale(X)

X_train, y_train = resample(X, y, n_samples=5000)

print("fit")

regressor = SVR(verbose=10)

gamma_range =  np.logspace(-5, 2, 10)
train_scores, valid_scores = validation_curve(regressor, X_train, y_train, "gamma", gamma_range, n_jobs=-1, scoring=rmsle_scorer)
valid_scores = [np.mean(s) for s in valid_scores]

# Take the alpha giving the highest validation score, and test it on test set
best_gamma = gamma_range[np.nanargmax(valid_scores)]
print("best gamma:", best_gamma)
regressor.set_params(gamma=best_gamma)

X_train, y_train = resample(X, y, n_samples=20000)
regressor.fit(X_train, y_train)

print("test")
# Since we can't load the whole dataset, do batch testing
batch_size = 5000
X_test, y_test = resample(X, y, n_samples=100000)
y_pred = np.ndarray((0,))
for i in range(0, X_test.shape[0], batch_size):
    print(i)
    y_pred = np.hstack((y_pred, regressor.predict(X_test[i: i + batch_size])))
print("RMSLE =", root_mean_squared_log_error(y_test, y_pred))

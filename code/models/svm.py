# Here import your model
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
from helpers import root_mean_squared_log_error

dataset = pd.read_csv('data/train_merged.csv')

# Subsamples with 15000 samples
dataset = dataset.sample(500000)

# Extract the objective values
y = dataset['trip_duration_in_minutes'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

means = np.nanmean(X, axis=0)
nan_locations = np.where(np.isnan(X))
X[nan_locations] = np.take(means, nan_locations[1])

# Normalize X
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

best_gamma = 0.01 # Found by grid search
regressor = SVR(gamma=best_gamma, verbose=10)
regressor.fit(X_train, y_train)
score = root_mean_squared_log_error(y_test, regressor.predict(X_test))
print("Root mean squared log error({}) = {}".format(best_gamma, score))

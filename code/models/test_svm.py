# Here import your model
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
import pandas as pd
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

# Normalize X
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
n_train = 5000
n_test = 1000
train_indices = np.random.choice(X_train.shape[0], n_train, replace=False)
test_indices = np.random.choice(X_test.shape[0], n_test, replace=False)
X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]


regressor = SVR()
regressor.set_params(gamma=0.0233)
regressor.fit(X_train, y_train)
print("Root mean squared error =", metrics.mean_squared_error(y_test, regressor.predict(X_test)) ** 0.5)

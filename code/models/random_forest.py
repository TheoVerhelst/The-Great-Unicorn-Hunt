from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve, train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import root_mean_squared_log_error

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

# Normalize X
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

regressor = RandomForestRegressor(n_estimators=200, n_jobs=-1)
regressor.fit(X_train, y_train)
print("RMSLE({}) = {}".format(n, root_mean_squared_log_error(regressor, X_test, y_test)))

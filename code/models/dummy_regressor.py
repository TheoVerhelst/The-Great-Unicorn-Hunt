from sklearn.model_selection import train_test_split
from sklearn import preprocessing, dummy, metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
print(dataset.columns.values)
X = dataset.values

# Normalize X
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

regressor = dummy.DummyRegressor()
regressor.fit(X_train, y_train)

print("Root mean squared error =", metrics.mean_squared_error(y_test, regressor.predict(X_test)) ** 0.5)

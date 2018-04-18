# Here import your model
from sklearn import linear_model

from sklearn.model_selection import validation_curve, train_test_split
from sklearn import preprocessing, metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset.values

# Normalize X
X = preprocessing.scale(X)

regressor = linear_model.Ridge()
alpha_range =  np.logspace(-5, 3, 20)
train_scores, valid_scores = validation_curve(regressor, X, y, "alpha", alpha_range, scoring="neg_mean_squared_error", n_jobs=-1)
train_scores = [np.mean(s) for s in train_scores]
valid_scores = [np.mean(s) for s in valid_scores]

plt.plot(alpha_range, train_scores)
plt.plot(alpha_range, valid_scores)
plt.xscale("log")
plt.show()

# Take the alpha giving the highest validation score, and test it on test set
best_alpha = alpha_range[np.nanargmax(valid_scores)]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
regressor.set_params(alpha=best_alpha)
regressor.fit(X_train, y_train)
print("best alpha =", alpha_range[np.nanargmax(valid_scores)])
print("Root mean squared error =", metrics.mean_squared_error(y_test, regressor.predict(X_test)) ** 0.5)

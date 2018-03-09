# Here import your model
from sklearn import linear_model

from sklearn.model_selection import validation_curve
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"]
# Add bias unit
dataset['bias_unit'] = 1
X = dataset.values

# Normalize X
X = preprocessing.scale(X)

regressor = linear_model.Ridge()
alpha_range =  np.logspace(-2, 10, 20)
train_scores, valid_scores = validation_curve(regressor, X, y, "alpha", alpha_range, n_jobs=-1)

print(alpha_range)
print([np.mean(s) for s in train_scores])
print([np.mean(s) for s in valid_scores])
plt.plot(alpha_range, [np.mean(s) for s in train_scores])
plt.plot(alpha_range, [np.mean(s) for s in valid_scores])
plt.xscale("log")
plt.show()

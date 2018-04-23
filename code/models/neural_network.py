from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
from helpers import root_mean_squared_log_error

dataset = pd.read_csv('data/train_merged.csv')

# Subsample the dataset, for quick learning tests

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"]
X = dataset

# Normalize X
X = (X - X.mean()) / X.std()
mean_y = y.mean()
std_y = y.std()
y = (y - mean_y) / std_y
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Feature columns describe how to use the input.
feature_columns = [tf.feature_column.numeric_column(key=key) for key in list(X)]
estimator = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[256, 128, 64])


def train_input_fn(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    return dataset.shuffle().batch(batch_size).repeat()

def test_input_fn(X):
    dataset = tf.data.Dataset.from_tensor_slices((dict(X),))
    return dataset.shuffle().batch(10)

batch_size = 1000
print("*"*30)
print("TRAINING...")
print("*"*30)
estimator.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size), steps=500)
print("*"*30)
print("TESTING...")
print("*"*30)
predictions = estimator.predict(input_fn=lambda:test_input_fn(X_test))
y_pred = np.array([e["predictions"][0] for e in predictions])
print(y_pred)
print("*"*30)
print(y_test)

y_test = y_test * std_y + mean_y
y_pred = y_pred * std_y + mean_y

print("RMSLE =", root_mean_squared_log_error(y_test, y_pred))

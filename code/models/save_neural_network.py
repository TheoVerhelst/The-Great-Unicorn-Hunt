from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from helpers import root_mean_squared_log_error, rmsle_scorer
from sklearn.externals import joblib
import numpy as np


output_filename = "data/submission_nn.csv"
model_filename = "serialized_neural_network.pkl"

train_frame = pd.read_csv('data/train_merged.csv')
test_frame = pd.read_csv('data/test_merged.csv')

# Extract the objective values
y_train = train_frame['trip_duration'].values
# Delete irrelevant columns in both set
del train_frame['trip_duration'], train_frame["id"], train_frame["trip_duration_in_minutes"]
# Save the test id for the output file, and remove them from test set
test_ids = test_frame["id"]
del test_frame['id']

X_train = train_frame.values
X_test = test_frame.values

means = np.nanmean(X_train, axis=0)
nan_locations = np.where(np.isnan(X_train))
X_train[nan_locations] = np.take(means, nan_locations[1])

means = np.nanmean(X_test, axis=0)
nan_locations = np.where(np.isnan(X_test))
X_test[nan_locations] = np.take(means, nan_locations[1])

# Normalize X_train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = MLPRegressor(solver='adam',
                     hidden_layer_sizes=(64,64,64), random_state=1,verbose=1, alpha=10 , max_iter=300, learning_rate_init=0.001)

print("Training the neural network")
regressor.fit(X_train, y_train)
print("Saving model to", model_filename)
joblib.dump(regressor, model_filename)
print("Testing on test set")
y_pred = regressor.predict(X_test)
print("Saving result to", output_filename)
output_frame = pd.concat([test_ids, pd.Series(y_pred)], axis=1, keys=["id", "trip_duration"])
output_frame.to_csv(output_filename, index=False)

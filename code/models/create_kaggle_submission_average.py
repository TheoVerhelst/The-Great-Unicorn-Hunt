from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

output_filename = "data/submission_average.csv"
model_filename_rf = "serialized_random_forest.pkl"
model_filename_nn = "serialized_neural_network.pkl"
model_filename_xgb = "serialized_xgboost.pkl"

# Load the training set for the sole purpose of getting its normalizing factors,
# in order to later apply them to test set
print("Loading training set to get its normalizing factors")
train_frame = pd.read_csv('data/train_merged.csv')
# Delete irrelevant columns in both set
del train_frame['trip_duration'], train_frame["id"], train_frame["trip_duration_in_minutes"]
# Normalize the training set
scaler = StandardScaler()
X_train = scaler.fit_transform(train_frame.values)

print("Loading test set")
test_frame = pd.read_csv('data/test_merged.csv')
# Save the test id for the output file, and remove them from test set
test_ids = test_frame["id"]
del test_frame['id']
X_test = test_frame.values

means = np.nanmean(X_test, axis=0)
nan_locations = np.where(np.isnan(X_test))
X_test[nan_locations] = np.take(means, nan_locations[1])

# Normalize the test set
print("Normalizing test set")
X_test = scaler.transform(X_test)

print("Loading model from", model_filename_rf, model_filename_nn, model_filename_xgb)
regressor_rf = joblib.load(model_filename_rf)
regressor_nn = joblib.load(model_filename_nn)
regressor_xgb = joblib.load(model_filename_xgb)


print("Testing on test set")
y_pred_rf = regressor_rf.predict(X_test)
y_pred_nn = regressor_nn.predict(X_test)
y_pred_xgb = regressor_xgb.predict(X_test)

y_pred = (np.array(y_pred_rf) + np.array(y_pred_nn) + np.array(y_pred_xgb)) / 3

print("Saving result to", output_filename)
output_frame = pd.concat([test_ids, pd.Series(y_pred)], axis=1, keys=["id", "trip_duration"])
output_frame.to_csv(output_filename, index=False)

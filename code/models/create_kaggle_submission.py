from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb


output_filename = "data/submission_xgb2.csv"
model_filename = "serialized_xgb_package.pkl"

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

dtest = xgb.DMatrix(X_test)
dtest.feature_names = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_minute', 'pickup_second', 'precipit_mm', 'straight_distance', 'bearing', 'manhattan_distance', 'trig_distance', 'osrm_trip_duration', 'turns', 'intersections', 'distance']

print("Loading model from", model_filename)
regressor = joblib.load(model_filename)
print("Testing on test set")
y_pred = regressor.predict(dtest)
print("Saving result to", output_filename)
output_frame = pd.concat([test_ids, pd.Series(y_pred)], axis=1, keys=["id", "trip_duration"])
output_frame.to_csv(output_filename, index=False)

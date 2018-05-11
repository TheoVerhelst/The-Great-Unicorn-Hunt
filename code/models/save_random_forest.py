from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np

output_filename = "data/submission.csv"
model_filename = "serialized_random_forest.pkl"

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
X_train = preprocessing.scale(X_train)

regressor = RandomForestRegressor(n_estimators=125, n_jobs=-1, verbose=10,
        bootstrap= False, max_depth=None, max_features=10, min_samples_leaf=3, min_samples_split=3)
print("Training the random forest")
regressor.fit(X_train, y_train)
print("Saving model to", model_filename)
joblib.dump(regressor, model_filename)
print("Testing on test set")
y_pred = regressor.predict(X_test)
print("Saving result to", output_filename)
output_frame = pd.concat([test_ids, pd.Series(y_pred)], axis=1, keys=["id", "trip_duration"])
output_frame.to_csv(output_filename, index=False)

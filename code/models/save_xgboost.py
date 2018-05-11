from sklearn import preprocessing
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import xgboost as xgb


output_filename = "data/submission_xgb.csv"
model_filename = "serialized_xgb.pkl"

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

# Normalize X_train
X_train = preprocessing.scale(X_train)

means = np.nanmean(X_test, axis=0)
nan_locations = np.where(np.isnan(X_test))
X_test[nan_locations] = np.take(means, nan_locations[1])

# fit XGBoost to training set
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train')]



#run XGB model
xgb_pars = {'min_child_weight': 9,      # min sum of instance `qweight needed in a child
            'eta': 0.05,                 # learning rate
            'colsample_bytree': 0.9,    # subsample ration of column when constructing each tree
            'max_depth': 9,             # max tree depth
            'subsample': 0.9,           # subsample ratio of the training instance
            'lambda': 1.2,               # l2 regularisation term on weights
            'nthread': -1,
            'booster' : 'gbtree',       # gbtree, gblinear or dart ----->   gbtree ~> dart >> gblinear
            'silent': 1,                # bool: print messages while running?
            'eval_metric': 'rmse',
            'objective': 'reg:linear',
            'gamma': 1,               # min loss reduction req to make further partition on leaf node (larger value = more conservative model?)
            'alpha': 0                  # l1 regularisation term on weights (inrease makes model more conservative?)}
            }


print("Training the booster")
model = xgb.train(xgb_pars, dtrain, 1000, watchlist, early_stopping_rounds=3, maximize=False, verbose_eval=2)
print("Saving model to", model_filename)
joblib.dump(model, model_filename)
print("Testing on test set")
y_pred = model.predict(dtest)
y_pred[y_pred < 0] = 0
print("Saving result to", output_filename)
output_frame = pd.concat([test_ids, pd.Series(y_pred)], axis=1, keys=["id", "trip_duration"])
output_frame.to_csv(output_filename, index=False)

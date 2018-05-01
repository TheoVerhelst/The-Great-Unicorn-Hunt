# XGBoost model

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from helpers import root_mean_squared_log_error

# import prepared dataset
dataset = pd.read_csv('data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"]
X = dataset.values

# Split dataset into tr and tst sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# fit XGBoost to training set
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]

#run XGB model
xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9,
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)

# final error metric
print("RMSLE =", root_mean_squared_log_error(y_test, model.predict(dtest)))
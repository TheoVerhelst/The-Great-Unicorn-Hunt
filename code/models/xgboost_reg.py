# XGBoost model

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from helpers import timer,root_mean_squared_log_error_minutes

# import prepared dataset
dataset = pd.read_csv('data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"],dataset['trip_duration_in_minutes']
X = dataset.values

# Split dataset into tr and tst sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# fit XGBoost to training set
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]



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
            'gamma': 0,               # min loss reduction req to make further partition on leaf node (larger value = more conservative model?)
            'alpha': 1000                  # l1 regularisation term on weights (inrease makes model more conservative?)}
            }

start = timer()#                     V number of iterations
model = xgb.train(xgb_pars, dtrain, 1000, watchlist, early_stopping_rounds=2, maximize=False, verbose_eval=1)
timer(start)
    #print('Modeling RMSLE %.5f' % model.best_score)

y_pred = model.predict(dtest)
y_pred[y_pred < 0] = 0 # Some predictions are negative, that screws up the log

    # final error metric
print("RMSLE = ", metrics.mean_squared_error(np.log(y_test + 1), np.log(y_pred + 1)) ** 0.5)
y_pred=np.round(y_pred/60)*60//60
y_test=np.round(y_test/60)*60//60
print("RMSLE for classification = ", metrics.mean_squared_error(np.log(y_test + 1), np.log(y_pred + 1)) ** 0.5)
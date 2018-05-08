# XGBoost model

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

# import prepared dataset
dataset = pd.read_csv('../../data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"],  dataset["trip_duration_in_minutes"]
X = dataset.values

# Split dataset into tr and tst sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# fit XGBoost to training set
#dtrain = xgb.DMatrix(X_train, label=y_train)
#dvalid = xgb.DMatrix(X_test, label=y_test)
#watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
 #           'max_depth': 6,
#'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
#'eval_metric': 'rmse', 'objective': 'reg:linear'}

# specify parameters and distributions to sample from 
regressor =XGBRegressor()

xgb_param = { 'booster' : ['gbtree'],
              'n_estimators':[75,100,125,150, 250],
              'eta': [0.05, 0.1, 0.3],
              'gamma':[0,0.2,0.4],
              'min_child_weight': [1,2],
              'max_depth': [ 6,7, 8,9,10],
              'subsample': [0.8,0.9, 1.0],
              'colsample_bytree': [0.9, 1.0],
              }
xgb_grid= GridSearchCV(regressor, xgb_param, n_jobs=8, 
                   cv=KFold(n_splits=5, shuffle=True), 
                   scoring='neg_mean_squared_log_error',
                   verbose=2,)
#Fit the model
xgb_grid.fit(X_train, y_train)
 
#trust your CV!
best_parameters, score, _ = max(xgb_grid.grid_scores_, key=lambda x: x[1])
print('RMSLE score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))


#Training the xgb boost model                       
#model = xgb.train(xgb_pars, dtrain, 100, watchlist, early_stopping_rounds=2,
     # maximize=False, verbose_eval=1)

#model.save_model('xgb1.model')
#bst.dump_model('xgbdump.raw.txt', 'xgb_featmap.txt')

print("Best Score:")
print(xgb_grid.best_score_)
print("Best Parameters:")
print(xgb_grid.best_params_)

print("RMSLE = {}".format( root_mean_squared_log_error(regressor, X_test, y_test)))


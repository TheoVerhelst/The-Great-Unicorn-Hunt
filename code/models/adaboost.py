# Here import your model
from sklearn import linear_model

from sklearn.model_selection import validation_curve,cross_val_score,train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
import pandas as pd
from time import time
#import matplotlib.pyplot as plt
import numpy as np
from helpers import *

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"],dataset['trip_duration_in_minutes']
X = dataset.values

# Normalize X
X = preprocessing.scale(X)
rng = np.random.RandomState(1)
#regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                          n_estimators=200, random_state=rng)
regr = AdaBoostRegressor(RandomForestRegressor(max_depth=4,n_jobs=-1),n_estimators=200, random_state=rng)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
print("starting fit")
start=time()
regr.fit(X_train,y_train);
end=time()
print(end-start,"seconds")

print("RMSLE =", root_mean_squared_log_error(regr, X_test, y_test))

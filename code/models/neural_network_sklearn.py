# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:24:12 2018

@author: GUILHERME
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import root_mean_squared_log_error, rmsle_scorer
from time import time
from scipy.stats import randint as sp_randint
from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV


dataset = pd.read_csv('../../data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset["trip_duration_in_minutes"], dataset["pickup_year"], dataset["store_and_fwd_flag"]
X = dataset.values

# Normalize X
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#regressor = MLPRegressor(solver='adam',
#                     hidden_layer_sizes=(64,64,64),early_stopping= False, random_state=1, verbose=1, alpha=100 , max_iter=300, tol=0.00001) 
regressor = MLPRegressor(solver='adam',
                     hidden_layer_sizes=(64,64,64), random_state=1,verbose=1, alpha=10 , max_iter=300, learning_rate_init=0.001) 
regressor.fit(X_train, y_train)

#gs = GridSearchCV(regressor,n_jobs=-1,scoring= rmsle_scorer, verbose=1, param_grid={
#    'alpha':[0.1,1,10,100,1000,10000],
#    'early_stopping':[200,300],
#    'activation': ['relu', 'tanh']})
    
#gs = GridSearchCV(nn, param_grid={
#    'learning_rate': [0.05, 0.01, 0.005, 0.001],
#    'hidden0__units': [4, 8, 12],
#    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
#grid_result=gs.fit(X_train, y_train)



print("RMSLE =", root_mean_squared_log_error(y_test, regressor.predict(X_test)))
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))


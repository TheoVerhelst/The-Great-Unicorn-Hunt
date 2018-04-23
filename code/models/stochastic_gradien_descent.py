# Here import your model
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import validation_curve,cross_val_score,train_test_split,GridSearchCV
from sklearn import preprocessing
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import numpy as np
from helpers import *

parameters={'loss':( 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
            'penalty' : ('none', 'l2', 'l1', 'elasticnet')}

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"],dataset['trip_duration_in_minutes']
X = dataset.values

# Normalize X
X = preprocessing.scale(X)
rng = np.random.RandomState(1)
regressor = SGDRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# alpha_range =  np.logspace(-5, 3, 20)
# train_scores, valid_scores = validation_curve(regressor, X_train, y_train, "alpha", alpha_range, scoring="neg_mean_squared_error")
# train_scores = [np.mean(s) for s in train_scores]
# valid_scores = [np.mean(s) for s in valid_scores]
#
# plt.plot(alpha_range, train_scores)
# plt.plot(alpha_range, valid_scores)
# plt.xscale("log")
# plt.show()
# best_alpha = alpha_range[np.nanargmax(valid_scores)]
# best_alpha=6.951927961775606e-05
# regressor.set_params(alpha=best_alpha)
# regressor.fit(X_train, y_train)
# print("best alpha =", alpha_range[np.nanargmax(valid_scores)])
# print("Root mean squared error =", root_mean_squared_log_error(regressor, X_test, y_test))

clf = GridSearchCV(regressor, parameters,verbose=10)
#clf=regressor
clf.fit(X_train,y_train)
print("Root mean squared error =", root_mean_squared_log_error(clf, X_test, y_test))
root_mean_squared_log_error_minutes(clf, X_test, y_test)
print(clf.get_params())

# Here import your model
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import preprocessing
import pandas as pd
import numpy as np
from helpers import root_mean_squared_log_error, rmsle_scorer

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"], dataset['trip_duration_in_minutes'], dataset['osrm_trip_duration']
X = dataset.values

# Normalize X
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

regressor = SGDRegressor()
parameters={'loss':( 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
            'penalty' : ('none', 'l2', 'l1', 'elasticnet')}

clf = GridSearchCV(regressor, parameters, verbose=10, scoring=rmsle_scorer)
clf.fit(X_train,y_train)

print("Root mean squared error =", root_mean_squared_log_error(y_test, clf.predict(X_test)))
print(clf.get_params())

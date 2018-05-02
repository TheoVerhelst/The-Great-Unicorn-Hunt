import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from time import time
from helpers import root_mean_squared_log_error

trainSet= pd.read_csv("data/train_merged.csv")
columns=["distance","trip_duration"]
y= trainSet["trip_duration_in_minutes"]
del trainSet["trip_duration_in_minutes"], trainSet["trip_duration"], trainSet["id"]

X=trainSet.values

# Normalize X
X = preprocessing.scale(X)
regressor = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
regressor.fit(X_train,y_train)
print("time : ",time()-start)
print("RMSLE =", root_mean_squared_log_error(y_test, regressor.predict(X_test)))

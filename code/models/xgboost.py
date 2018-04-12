# XGBoost model

# import libraries
#%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [16, 10]

# import prepared dataset
dataset = pd.read_csv('data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"], dataset['distance']
#X = preprocessing.scale(dataset.values)
X = dataset.values

# visualise data + obtain data summary
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#dataset.head()
#dataset.describe()
#dataset.info()

# Split dataset into tr and tst sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# fit XGBoost to 
classifier = XGBClassifier()
classifier.fit(X_train, y_train)        
# !! this currently spits out the following error: 'WindowsError: [Error -529697949] Windows Error 0xE06D7363'
# trying to fix this but at least the rest of the code is available

# Predicting test set resuts
y_pred = classifier.predict(X_test)

# Build confusion matrix
cm = confusion_matrix(y_test, y_pred)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.std()
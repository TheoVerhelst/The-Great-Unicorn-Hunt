import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from time import time
clf = DecisionTreeClassifier(random_state=0)
trainSet= pd.read_csv("../../data/train_merged.csv")
columns=["distance","trip_duration"]
y= trainSet["trip_duration_in_minutes"]
del trainSet["trip_duration_in_minutes"], trainSet["trip_duration"], trainSet["id"]

X=trainSet.values

# Normalize X
X = preprocessing.scale(X)
print("starting crossvalidation")
start=time()
scores=cross_val_score(clf,X,y,cv=10)
end=time()
print(end-start,"seconds")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

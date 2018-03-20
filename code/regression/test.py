from sklearn import svm
import pandas as pd

trainSet= pd.read_csv("../../data/train/train_distances_features.csv")

print(trainSet.head(3))
columns=["distance","trip_duration"]

features=trainSet[list(columns)].values
target=trainSet["trip_duration"].values

clf=svm.SVR(max_iter=10)
print(clf.fit(features, target))

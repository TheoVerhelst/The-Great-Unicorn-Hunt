from sklearn.model_selection import train_test_split
import pandas as pd
from tpot import TPOTRegressor


dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"]

# Normalize X
X = dataset.values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2,
        warm_start=True, memory="data/tpot_cache", n_jobs=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')

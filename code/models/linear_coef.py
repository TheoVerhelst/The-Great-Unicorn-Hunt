# Here import your model
from sklearn import linear_model

from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/train_merged.csv')

# Extract the objective values
y = dataset['trip_duration'].values
# Delete irrelevant columns in training set
del dataset['trip_duration'], dataset["id"]
# Add bias unit
dataset['bias_unit'] = 1

# Normalize X
X = preprocessing.scale(dataset.values)

alpha = 14 # Found by grid search
regressor = linear_model.Ridge(alpha=alpha)
regressor.fit(X, y)

print("Score:", regressor.score(X, y))

plt.figure()
plt.xticks(rotation='vertical')
plt.bar(list(dataset), [abs(c) for c in regressor.coef_])
plt.subplots_adjust(bottom=0.26)
plt.yscale('log')
plt.title("Coeffs of linear ridge regression ($\\alpha = 14$)")
plt.show()

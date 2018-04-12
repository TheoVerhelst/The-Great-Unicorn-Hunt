from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"]
X = dataset.values

X = preprocessing.scale(X)

pca = PCA(n_components=2)
transformed = pca.fit_transform(X)

plt.figure()
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.26)
# Apply absolute value since we are interested in importance, not sign of features
plt.bar(list(dataset), abs(pca.components_[0]))
plt.title("First PCA component values")
plt.figure()
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.26)
plt.bar(list(dataset), abs(pca.components_[1]))
plt.title("Second PCA component values")

heatmap, xedges, yedges = np.histogram2d(transformed[:, 0], transformed[:, 1], weights=y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.figure()
plt.clf()
plt.imshow(heatmap, extent=extent, cmap='hot')
plt.show()

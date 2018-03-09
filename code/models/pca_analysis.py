from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/train_merged.csv')
y = dataset['trip_duration'].values
del dataset['trip_duration'], dataset["id"], dataset['distance']
#X = preprocessing.scale(dataset.values)
X = dataset.values


pca = PCA(n_components=2)
transformed = pca.fit_transform(X)
heatmap, xedges, yedges = np.histogram2d(transformed[:, 0], transformed[:, 1], weights=y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent, cmap='hot')
plt.show()

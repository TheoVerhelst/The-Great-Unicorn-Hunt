import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/train_merged.csv')
dataset = dataset[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]]

long = np.concatenate((dataset["pickup_longitude"].values, dataset["dropoff_longitude"].values))
lat = np.concatenate((dataset["pickup_latitude"].values, dataset["dropoff_latitude"].values))

sns.jointplot(x=long, y=lat, kind='hex', stat_func=None)
plt.show()

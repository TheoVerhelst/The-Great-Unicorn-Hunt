# This file studies the correlation between the distance and time predicted by OSRM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/train_merged.csv')
dataset = dataset[["distance", "osrm_trip_duration"]]

graph = sns.pairplot(dataset, plot_kws={"s": 1, "linewidth":1, "color":"b", "edgecolor":"b"}, markers=".")
plt.show()

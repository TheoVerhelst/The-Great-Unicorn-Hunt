import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/train_merged.csv')

del dataset["id"], dataset["trip_duration_in_minutes"]

# Take a subset of the dataset, to be lighter on graphics (might crash with the
# whole dataset).
n = 1000
dataset = dataset.sample(n)

corr = dataset.corr()

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True))
"""
graph = sns.pairplot(dataset, plot_kws={"s": 5, "linewidth":1, "color":"b", "edgecolor":"b"}, markers=".")
# Remove axes number labels (we're not really interested in actual numbers on these)
graph.set(xticklabels=[], yticklabels=[])
# Set margins
plt.subplots_adjust(bottom=0.15, left=0.1)
# Rotate variable names
for ax in graph.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation=50)
    ax.set_ylabel(ax.get_ylabel(), rotation="horizontal")
"""
plt.show()

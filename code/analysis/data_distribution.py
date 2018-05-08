import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/train_merged.csv')
"""
plt.figure()
ax = sns.distplot(dataset["dropoff_longitude"], label="Dropoff longitude", axlabel="Longitude", kde=True)
sns.distplot(dataset["pickup_longitude"], ax=ax, label="Pickup longitude", axlabel="Longitude", kde=True)
ax.set_ylabel("Density")
plt.legend()

plt.figure()
ax = sns.distplot(dataset["dropoff_latitude"], label="Dropoff latitude", axlabel="Latitude", kde=True)
sns.distplot(dataset["pickup_latitude"], ax=ax, label="Pickup latitude", axlabel="Latitude", kde=True)
ax.set_ylabel("Density")
plt.legend()

plt.figure()
ax = sns.distplot(dataset["trip_duration"], axlabel="Trip duration (min)", norm_hist=False)
ax.set_ylabel("Density")

"""
plt.figure()
ax = sns.barplot(x="passenger_count", y="passenger_count", data=dataset, estimator=lambda x: len(x) / len(dataset))
ax.set_ylabel("Density")
ax.set_xlabel("Passenger count")

"""
### Test VS training set distribution comparison
del dataset

train = pd.read_csv('data/train_merged.csv')

test = pd.read_csv('data/test_merged.csv')

xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

test = test[(test.pickup_longitude > xlim[0]) & (test.pickup_longitude < xlim[1])]
test = test[(test.dropoff_longitude > xlim[0]) & (test.dropoff_longitude < xlim[1])]
test = test[(test.pickup_latitude > ylim[0]) & (test.pickup_latitude < ylim[1])]
test = test[(test.dropoff_latitude > ylim[0]) & (test.dropoff_latitude < ylim[1])]

plt.figure()
ax = sns.distplot(train["dropoff_longitude"], label="Training set", axlabel="Dropoff longitude")
sns.distplot(test["dropoff_longitude"], ax=ax, label="Test set", axlabel="Dropoff longitude")
ax.set_ylabel("Density")
plt.legend()

plt.figure()
ax = sns.distplot(train["dropoff_latitude"], label="Training set", axlabel="Dropoff latitude")
sns.distplot(test["dropoff_latitude"], ax=ax, label="Test set", axlabel="Dropoff latitude")
ax.set_ylabel("Density")
plt.legend()
"""

plt.show()

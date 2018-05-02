# Data Visualisation

# import libraries
import pandas as pd
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]

# import prepared dataset
dataset = pd.read_csv('data/train_merged.csv')
y = dataset[['trip_duration']].values
X = dataset.drop(['trip_duration', 'id'], axis=1)

# trip duration histogram
plt.hist(dataset['trip_duration'].values, bins=100)
plt.xlabel('Trip Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(b=1)
plt.show()

# trip duration log(histogram)
dataset['log(Trip Duration)'] = np.log(dataset['trip_duration'].values + 1)
plt.hist(dataset['log_trip_duration'].values, bins=100)
plt.xlabel('log(Trip Duration)')
plt.ylabel('number of train records')
plt.grid(b=1)
plt.show()
sns.distplot(dataset["log(Trip Duration)"], bins =100)

# Time per vendor
import warnings
warnings.filterwarnings("ignore")
plot_vendor = dataset.groupby('vendor_id')['trip_duration'].mean()
plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=800)
plt.ylim(ymax=840)
sns.barplot(plot_vendor.index,plot_vendor.values)
plt.title('Average Trip Time per Vendor')
plt.grid(b=1)
plt.legend(loc=0)
plt.ylabel('Time Duration(seconds)')

# Time per store_flag
snwflag = dataset.groupby('store_and_fwd_flag')['trip_duration'].mean()
plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Average Trip Time per store_and_fwd_flag')
plt.legend(loc=0)
plt.ylabel('Trip Duration (seconds)')
plt.grid(b=1)
sns.barplot(snwflag.index,snwflag.values)

# Time per passenger count
pc = dataset.groupby('passenger_count')['trip_duration'].mean()
plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Average Trip Time per Passenger Count')
plt.legend(loc=0)
plt.grid(b=1)
plt.ylabel('Trip Duration (seconds)')
sns.barplot(pc.index,pc.values)

# Heatmap
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)
ax.scatter(X['pickup_longitude'].values[:100000], X['pickup_latitude'].values[:100000], color='blue', s=1, alpha=0.6)
fig.suptitle('Trip Heatmap')
ax.legend(loc=0)
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.grid(b=1)
plt.show()

# Neighbourhoods with Kmeans
coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values, X[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
X.loc[:, 'pickup_cluster'] = kmeans.predict(X[['pickup_latitude', 'pickup_longitude']])
X.loc[:, 'dropoff_cluster'] = kmeans.predict(X[['dropoff_latitude', 'dropoff_longitude']])
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(X.pickup_longitude.values[:500000], X.pickup_latitude.values[:500000], s=10, lw=0, c=X.pickup_cluster[:500000].values, cmap='hot', alpha=1)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid()
plt.show()

# Augmenting data with neighbourhoods
vendor_ = pd.get_dummies(dataset['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_ = pd.get_dummies(dataset['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_ = pd.get_dummies(dataset['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup = pd.get_dummies(X['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff = pd.get_dummies(X['dropoff_cluster'], prefix='d', prefix_sep='_')

X = pd.concat([X, cluster_pickup, cluster_dropoff])
X = dataset.drop(['vendor_id', 'passenger_count', 'store_and_fwd_flag', ], axis=1)
X.to_csv("train_merged_with_clusters",index=False)
X2 = pd.concat([X, vendor_, passenger_count_, store_and_fwd_flag_)
X2.to_csv("train_merged_with_cluster_dummy",index=False)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

matplotlib.use('TkAgg')  # using matplotlib in the backend

# open the file
train = pd.read_csv('data/train.csv')
rain = pd.read_csv('data/rain.csv')

"""
print("**********")
print("Description of the dataset")
print(train.describe())
print(rain.describe())
print("**********")

# Confirm if there are any nan values
print(train.isnull().sum())

# Explore the data
print(train['vendor_id'].unique())  # Unique values 1 and 2
print(train['passenger_count'].unique())  # Values range from 0-9
print(train['pickup_longitude'].unique())
"""

# Clean up trip duration in train data
# We see some absurd trip duration in the training set. So we clean up all
# values that are greater that 2 standard deviations
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2 * s]
train = train[train['trip_duration'] >= m - 2 * s]

# Convert the pickup and dropoff times to datetime objects
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'].str.strip(), format='%Y-%m-%d %H:%M:%S')
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'].str.strip(), format='%Y-%m-%d %H:%M:%S')
rain['datetime'] = pd.to_datetime(rain['datetime'].str.strip(), format='%d/%m/%Y %H:%M')

# Here we decide to use the wheather data from PICKUP TIME, we could also choose
# DROPOFF TIME, this should be decided at validation time
train['pickup_datetime_hour'] = train['pickup_datetime'].map(lambda timestamp:timestamp.replace(minute=0, second=0))

# Augmenting data - matching rain data to date and time

train = pd.merge(train, rain, left_on='pickup_datetime_hour', right_on='datetime', validate='many_to_one')

# Iterate over all time components, and put each of them in a new column
# A bit of dirty reflection here, we use the year, month, ... atributes of datetime objects
for attribute in ("year", "month", "day", "hour", "minute", "second"):
    train['pickup_' + attribute] = getattr(train['pickup_datetime'].dt, attribute)
    train['dropoff_' + attribute] = getattr(train['dropoff_datetime'].dt, attribute)

# Erase the datetime objects, we don't, need them anymore
del train['datetime'], train['pickup_datetime'], train['dropoff_datetime']

# Remove not features columns
del train['id'], train['pickup_datetime_hour']

# Remove rides to and from far away areas
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

train = train[(train.pickup_longitude > xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude > xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude > ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude > ylim[0]) & (train.dropoff_latitude < ylim[1])]

# Plot rides
longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)
latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)
plt.figure(figsize=(10, 10))
plt.plot(longitude, latitude, '.', alpha=0.4, markersize=0.05)
plt.show()

print(train)

# write dataframe into new csv file
train.to_csv('data/train_features.csv')

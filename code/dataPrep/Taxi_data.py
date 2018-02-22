import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # using matplotlib in the backend

# True to parse the training set, False to parse the test set instead
prepare_training_set = False

# open the file
if prepare_training_set:
    dataset = pd.read_csv('data/train.csv')
else:
    dataset = pd.read_csv('data/test.csv')

rain = pd.read_csv('data/rain.csv')

"""
print("**********")
print("Description of the dataset")
print(dataset.describe())
print(rain.describe())
print("**********")

# Confirm if there are any nan values
print(dataset.isnull().sum())

# Explore the data
print(dataset['vendor_id'].unique())  # Unique values 1 and 2
print(dataset['passenger_count'].unique())  # Values range from 0-9
print(dataset['pickup_longitude'].unique())
"""

if prepare_training_set:
    # Clean up trip duration in train data
    # We see some absurd trip duration in the training set. So we clean up all
    # values that are greater that 2 standard deviations
    m = np.mean(dataset['trip_duration'])
    s = np.std(dataset['trip_duration'])
    dataset = dataset[dataset['trip_duration'] <= m + 2 * s]
    dataset = dataset[dataset['trip_duration'] >= m - 2 * s]

# Convert the pickup and dropoff times to datetime objects
dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'].str.strip(), format='%Y-%m-%d %H:%M:%S')
rain['datetime'] = pd.to_datetime(rain['datetime'].str.strip(), format='%d/%m/%Y %H:%M')

# Here we decide to use the wheather data from PICKUP TIME, since at test time
# this is the only timestamp available (obviously we don't have dropoff time)
dataset['pickup_datetime_hour'] = dataset['pickup_datetime'].map(lambda timestamp:timestamp.replace(minute=0, second=0))

# Augmenting data - matching rain data to date and time
dataset = pd.merge(dataset, rain, left_on='pickup_datetime_hour', right_on='datetime', validate='many_to_one')

# Iterate over all time components, and put each of them in a new column
# A bit of dirty reflection here, we use the month, day, ... attributes of
# datetime objects
for attribute in ("month", "day", "hour", "minute", "second"):
    dataset['pickup_' + attribute] = getattr(dataset['pickup_datetime'].dt, attribute)

dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map(lambda flag: 1 if flag == 'Y' else 0)

# Erase the datetime objects, we don't, need them anymore
del dataset['datetime'], dataset['pickup_datetime']

if prepare_training_set:
    del dataset['dropoff_datetime']

# Remove not features columns
del dataset['id'], dataset['pickup_datetime_hour']

# Remove rides to and from far away areas
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

dataset = dataset[(dataset.pickup_longitude > xlim[0]) & (dataset.pickup_longitude < xlim[1])]
dataset = dataset[(dataset.dropoff_longitude > xlim[0]) & (dataset.dropoff_longitude < xlim[1])]
dataset = dataset[(dataset.pickup_latitude > ylim[0]) & (dataset.pickup_latitude < ylim[1])]
dataset = dataset[(dataset.dropoff_latitude > ylim[0]) & (dataset.dropoff_latitude < ylim[1])]

# Plot rides
longitude = list(dataset.pickup_longitude) + list(dataset.dropoff_longitude)
latitude = list(dataset.pickup_latitude) + list(dataset.dropoff_latitude)
plt.figure(figsize=(10, 10))
plt.plot(longitude, latitude, '.', alpha=0.4, markersize=0.05)
plt.show()

# write dataframe into new csv file
if prepare_training_set:
    dataset.to_csv('data/train_features.csv')
else:
    dataset.to_csv('data/test_features.csv')

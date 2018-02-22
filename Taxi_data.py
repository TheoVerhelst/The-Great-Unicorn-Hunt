import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

matplotlib.use('TkAgg')  # using matplotlib in the backend

# open the file
train = pd.read_csv('/Users/Gui/PycharmProjects/Taxi_Project/train 2.csv')
rain = pd.read_csv('/Users/Gui/PycharmProjects/Taxi_Project/rain.csv')


# give the first give elements
print(train.head())
print(train.describe())

# Confirm if there are any nan values
print(train.isnull().sum())
# print(test.isnull().sum())

# Explore the data

print(train['vendor_id'].unique())  # Unique values 1 and 2
print(train['passenger_count'].unique())  # Values range from 0-9
print(train['pickup_longitude'].unique())
# test['vendor_id'].unique() # Unique values 1 and 2
# test['passenger_count'].unique() #Values range from 0-9

# Clean up trip duration in train data
# We see that trip duration is 1 or 0 which is not possible. so we clean up all values that are greater that 2 standard deviations
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]

# separate date time into date and time, while adding to extra collumns
train['pickup_date'], train['pickup_time'] = train['pickup_datetime'].str.split(' ', 1).str
train['dropoff_date'], train['dropoff_time'] = train['dropoff_datetime'].str.split(' ', 1).str

train['pickup_year'], train['pickup_month'],train['pickup_day'] = train['pickup_date'].str.split('-').str
train['pickup_hour'], train['pickup_minute'], train['pickup_seconds']= train['pickup_time'].str.split(':').str
train['dropoff_year'], train['dropoff_month'],train['dropoff_day'] = train['dropoff_date'].str.split('-').str
train['dropoff_hour'], train['dropoff_minute'],train['dropoff_seconds'] = train['dropoff_time'].str.split(':').str



# Reformatting the time collumn of the weather(rain) data set in order to have only Hour instead of Hour:min
# this will help with matching hour of weather data to training data
rain['time'] =  pd.to_datetime(rain['time'].str.strip(), format='%H:%M')
rain['time'] = rain['time'].dt.hour

train['pickup_date'] =  pd.to_datetime(train['pickup_date'].str.strip(), format='%Y-%m-%d')
train['pickup_date'] = train['pickup_date'].dt.strftime('%d%b')

# delete collumns
# del train['pickup_datetime']
# del train['dropoff_datetime']

print(rain)
print(train)


# sort data by pickup time if desired
# train = train.sort_values(by='pickup_datetime')
# print(type(train['pickup_datetime']))

# augmenting data - matching rain data to date and time

# train["rain(mm)"] = ""
# print(df.head())
# print('YAYAYAYAYAYYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAY/n/n/n')
#
# for i in range(len(train)):         # iterate through all entries of Taxi
#    taxiDate = train.at[i, 'pickup_date']   # save the date for that entry
#    taxiTime = train.at[i, 'pickup_hour']   # save the time for that entry
#
#
#    for ii in range(0, len(rain):   # iterate through all entries of Rain
#        if taxiDate == rain.at[ii, 'date'] and taxiTime == rain.at[ii, ' time']:     # check if date and time match
#            rain_value = rain.at[ii, 'precipit_mm']
#            train.at[i, 'rain(mm)'] = rain
#            break
#
#
#
# print(train)

# Remove rides to and from far away areas
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

# xlim = [-74.04, -73.74]
# ylim = [40.57, 40.88]

train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]

# Plot rides
longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)
latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
plt.show()

loc_df = pd.DataFrame()
loc_df['longitude'] = longitude
loc_df['latitude'] = latitude

# way of localising and isolating only desired collumns (by name) and rows (by index)
df1 = df.loc[1:1048576, "pickup_datetime":"dropoff_datetime"]

# write dataframe into new csv file
train.to_csv('train_clean', sep='\t')

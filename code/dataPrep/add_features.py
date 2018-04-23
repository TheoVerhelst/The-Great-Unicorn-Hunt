import pandas as pd
import math
from math import sin, cos, sqrt, atan2, radians
import geopy.distance
#from geopy.distance import vincenty
def main(input_file, rain_file, output_file):
    """
    expected input_file : {train/test}_parsed.csv
    expected output_file: {train/test}_added_features.csv
    """
    dataset = pd.read_csv(input_file)
    rain = pd.read_csv(rain_file, skipinitialspace=True)

    # Convert the pickup and rain times to datetime objects
    dataset['pickup_datetime'] = pd.to_datetime(dict(year=dataset['pickup_year'],
            month=dataset['pickup_month'], day=dataset['pickup_day'], hour= dataset['pickup_hour']))
    rain['datetime'] = pd.to_datetime(rain['datetime'].str.strip(), format='%d/%m/%Y %H:%M')

    # Augmenting data - matching rain data to pickup time
    dataset = pd.merge(dataset, rain, left_on='pickup_datetime', right_on='datetime', validate='many_to_one')

    # Create other features here
    dataset["dow"] = dataset['pickup_datetime'].dt

    #Calculate distance between coordinates
    #Creating list to store values
    straight_dist_list = []
    compass_bearing_list=[]
    manhattan_dist_list=[]
    trig_dist_list=[]

    for index, row in dataset.iterrows():

        lat_pickup=row['pickup_latitude']
        lat_dropoff = row['dropoff_latitude']
        long_pickup = row['pickup_longitude']
        long_dropoff = row['dropoff_longitude']

        #using actual formula, however apparently less accurate than Vincenty distance used in geopy by 0.5%
        #R=6371
        # a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        # c = 2 * atan2(sqrt(a), sqrt(1 - a))
        #distance = R * c
        #straight_dist_list.append(distance)

        #Using geopy library, calculate the straight distance
        coords_1=(lat_pickup, long_pickup)
        coords_2=(lat_dropoff,long_dropoff)

        straight_dist_list.append(geopy.distance.VincentyDistance(coords_1, coords_2).km)

        #Calculate trig distance - pythagoras
        trig_dist_list.append(sqrt((lat_dropoff-lat_pickup)**2+(long_dropoff-long_pickup)** 2 ))

        #Convert to radians
        lat1 = math.radians(lat_pickup)
        lat2 = math.radians(lat_dropoff)
        long1=math.radians(long_pickup)
        long2 = math.radians(long_dropoff)
        dLong = long2 - long1
        dLat= lat2-lat1

        #Calculate the Manhattan Distance
        R=6371
        a = sin(dLat / 2) ** 2
        cLat = 2 * atan2(sqrt(a), sqrt(1 - a))
        latitudeDistance = R * cLat

        a = sin(dLong / 2) ** 2
        cLong = 2 * atan2(sqrt(a), sqrt(1 - a))
        longitudeDistance = R * cLong

        manhattan_dist_list.append(abs(latitudeDistance+longitudeDistance))

        #Calculate the bearing
        x = math.sin(dLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                               * math.cos(lat2) * math.cos(dLong))

        initial_bearing = math.atan2(x, y)

        # Now we have the initial bearing but math.atan2 return values
        # from -180° to + 180° which is not what we want for a compass bearing
        # The solution is to normalize the initial bearing as shown below
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        compass_bearing_list.append(compass_bearing)

    #Passing these lists into series
    se_bearing = pd.Series(compass_bearing_list)
    se_manhattan = pd.Series(manhattan_dist_list)
    se_dist = pd.Series(straight_dist_list)
    se_trig= pd.Series(trig_dist_list)

    #Passing them into the dataset as new collumns
    dataset['straight_distance'] = se_dist.values
    dataset['bearing'] = se_bearing.values
    dataset['manhattan_distance'] = se_manhattan.values
    dataset['trig_distance'] = se_trig.values

    # Keep only the id and the rain feature
    dataset = dataset[['id', 'precipit_mm','straight_distance','bearing','manhattan_distance', 'trig_distance']]

    # write dataframe into new csv file
    dataset.to_csv(output_file,index=False)

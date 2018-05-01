import urllib.request
import json
import pandas as pd
import numpy as np
# Import the dataset file paths
from prepare import original_file, distances_features_file

def getFastestRoute(pickupLong, pickupLat, dropoffLong, dropoffLat):

    url = 'http://127.0.0.1:5000/route/v1/driving/{0},{1};{2},{3}'.format(
    pickupLong, pickupLat, dropoffLong, dropoffLat)

    response = None
    try:
        response = urllib.request.urlopen(url).read().decode('utf-8')
    except urllib.error.HTTPError as e:
            return 0, 0

    response = json.loads(response)
    distance = response["routes"][0]["distance"]
    trip_duration = response["routes"][0]["duration"]
    return distance, trip_duration

def main(input_file, output_file):
    """
    expected input_file : {train/test}.csv
    expected output_file: {train/test}_distances.csv
    """
    dataset = pd.read_csv(input_file)
    # Use numpy.vectorize to call getFasterRoute on the whole column
    # And zip to assign both columns at once
    dataset["distance"], dataset["osrm_trip_duration"] = np.vectorize(getFastestRoute)(dataset["pickup_longitude"],
            dataset["pickup_latitude"], dataset["dropoff_longitude"], dataset["dropoff_latitude"])

    dataset = dataset[["distance", "id", "osrm_trip_duration"]]

    # write dataframe into new csv file
    dataset.to_csv(output_file, index=False)

if __name__ == "__main__":
    main(original_file, distances_features_file)

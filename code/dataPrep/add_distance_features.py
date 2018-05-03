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
    except urllib.error.HTTPError:
            return 0, 0

    response = json.loads(response)
    route = response["routes"][0]
    distance = route["distance"]
    trip_duration = route["duration"]

    stepsObject = route["legs"][0]["steps"]
    turns = len(stepsObject) - 2
    intersections = -2
    for item in stepsObject:
		intersections += len(item["intersections"])

    return distance, trip_duration, turns, intersections

def main(input_file, output_file):
    """
    expected input_file : {train/test}.csv
    expected output_file: {train/test}_distances.csv
    """
    dataset = pd.read_csv(input_file)
    # Use numpy.vectorize to call getFasterRoute on the whole column
    # And zip to assign both columns at once
    dataset["distance"], dataset["osrm_trip_duration"] , dataset["turns"] , dataset["intersections"] = \
            np.vectorize(getFastestRoute)(dataset["pickup_longitude"],
            dataset["pickup_latitude"], dataset["dropoff_longitude"], dataset["dropoff_latitude"])

    dataset = dataset[["distance", "id", "osrm_trip_duration", "turns", "intersections"]]

    # write dataframe into new csv file
    dataset.to_csv(output_file, index=False)

if __name__ == "__main__":
    main(original_file, distances_features_file)

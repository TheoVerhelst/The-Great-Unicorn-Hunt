import pandas as pd
import numpy as np

def main(input_file, output_file):
    dataset = pd.read_csv(input_file)

    if 'trip_duration' in dataset:
        # Clean up trip duration in train data
        # We see some absurd trip duration in the training set. So we clean up all
        # values that are greater that 2 standard deviations
        m = np.mean(dataset['trip_duration'])
        s = np.std(dataset['trip_duration'])
        dataset = dataset[dataset['trip_duration'] <= m + 2 * s]
        dataset = dataset[dataset['trip_duration'] >= m - 2 * s]

    # Remove rides to and from far away areas
    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]

    dataset = dataset[(dataset.pickup_longitude > xlim[0]) & (dataset.pickup_longitude < xlim[1])]
    dataset = dataset[(dataset.dropoff_longitude > xlim[0]) & (dataset.dropoff_longitude < xlim[1])]
    dataset = dataset[(dataset.pickup_latitude > ylim[0]) & (dataset.pickup_latitude < ylim[1])]
    dataset = dataset[(dataset.dropoff_latitude > ylim[0]) & (dataset.dropoff_latitude < ylim[1])]

    # write dataframe into new csv file
    dataset.to_csv(output_file)

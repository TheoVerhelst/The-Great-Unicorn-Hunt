import pandas as pd

def main(input_file, output_file,prefix="troll"):
    """
    expected input_file : {train/test}.csv
    expected output_file: {train/test}_parsed.csv
    """
    dataset = pd.read_csv(input_file)

    # Convert the pickup and dropoff times to datetime objects
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'].str.strip(), format='%Y-%m-%d %H:%M:%S')

    # Iterate over all time components, and put each of them in a new column
    # A bit of dirty reflection here, we use the month, day, ... attributes of
    # datetime objects
    for attribute in ("year", "month", "day", "hour", "minute", "second"):
        dataset['pickup_' + attribute] = getattr(dataset['pickup_datetime'].dt, attribute)

    # Convert flag 'N'/'Y' to 0/1
    dataset['store_and_fwd_flag'] = dataset['store_and_fwd_flag'].map(lambda flag: 1 if flag == 'Y' else 0)

    # Erase the datetime objects, we don't, need them anymore
    del dataset['pickup_datetime']

    # Remove useless columns
    if 'dropoff_datetime' in dataset:
        del dataset['dropoff_datetime']
    del dataset['pickup_year'] # This column is always 2016, not very informative

    if prefix=="train":
        dataset["trip_duration_in_minutes"] = round(dataset["trip_duration"]/60)*60//60

    # write dataframe into new csv file
    dataset.to_csv(output_file, index=False)

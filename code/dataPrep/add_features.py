import pandas as pd

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

    # Keep only the id and the rain feature
    dataset = dataset[['id', 'precipit_mm']]

    # write dataframe into new csv file
    dataset.to_csv(output_file,index=False)

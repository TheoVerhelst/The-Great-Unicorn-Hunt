import pandas as pd
import parse, clean, add_features

path = '../../data/'
# Change to 'test' to prepare the test dataset
prefix = 'train'
original_file           = path + prefix + '.csv'
parsed_file             = path + prefix + '_parsed.csv'
added_features_file     = path + prefix + '_added_features.csv'
distances_features_file = path + prefix + '_distances.csv'
merged_file             = path + prefix + '_merged.csv'
rain_dataset            = path + "rain.csv"

# Parse the original dataset
print("Parsing of original data...")
parse.main(original_file, parsed_file,prefix)
# Create features from the dataset
print("Creating additional features...")
add_features.main(parsed_file, rain_dataset, added_features_file)

# Merge the three files : parsed_file, added_features_file and
# distances_features_file into merged_file
print("Merging all features...")
parsed_df = pd.read_csv(parsed_file)
added_features_df = pd.read_csv(added_features_file)
distances_features_df = pd.read_csv(distances_features_file)
merged_df = pd.merge(parsed_df, pd.merge(added_features_df, distances_features_df, on='id'), on='id')
merged_df.to_csv(merged_file,index=False)

# Clean the final dataset
print("Cleaning data...")
clean.main(merged_file, merged_file)

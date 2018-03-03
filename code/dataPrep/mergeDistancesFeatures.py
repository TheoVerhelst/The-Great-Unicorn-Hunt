import pandas as pd

test = False
if test:
    distances = pd.read_csv('../../data/test/test_distances.csv')
    features = pd.read_csv('../../data/test/test_features_with_id.csv')
else:
    distances = pd.read_csv('../../data/train/train_distances.csv')
    features = pd.read_csv('../../data/train/train_features_with_id.csv')

finalDataset = pd.merge(distances, features, left_on='id', right_on='id')

if test:
    finalDataset.to_csv('../../data/test/test_distances_features.csv')
else:
    finalDataset.to_csv('../../data/train/train_distances_features.csv')

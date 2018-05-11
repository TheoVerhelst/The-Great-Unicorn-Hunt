import pandas as pd

output_filename = "data/submission_average.csv"

rf_pred = pd.read_csv("data/submission_rf_gui.csv")
nn_pred = pd.read_csv("data/submission_nn.csv")

nn_pred.loc[nn_pred.trip_duration < 0, 'trip_duration'] = 0

print("RF")
print(rf_pred.describe())
print("NN")
print(nn_pred.describe())

merged_pred = pd.DataFrame()
merged_pred["id"] = rf_pred["id"]
merged_pred["trip_duration"] = (rf_pred["trip_duration"] + nn_pred["trip_duration"]) / 2

merged_pred.to_csv(output_filename, index=False)

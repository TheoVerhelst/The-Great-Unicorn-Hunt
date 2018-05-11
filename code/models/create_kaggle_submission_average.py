import pandas as pd

output_filename = "data/submission_average.csv"

rf_pred = pd.read_csv("data/submission_rf.csv")
xgb_pred = pd.read_csv("data/submission_xgb.csv")
nn_pred = pd.read_csv("data/submission_nn.csv")

merged_pred = pd.DataFrame()
merged_pred["id"] = rf_pred["id"]
merged_pred["trip_duration"] = (pred_rf["trip_duration"] + pred_xgb["trip_duration"] + pred_nn["trip_duration"]) / 3

merged_pred.to_csv(output_filename, index=False)

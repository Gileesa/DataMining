# Comparison
import pandas as pd

rf_results = pd.read_csv("DataMining/csv_files/Classification/RandomForest/rf_results.csv")
xgb_results = pd.read_csv("DataMining/csv_files/Classification/XGBOOST/xgb_results.csv")

summary = pd.concat([rf_results, xgb_results], ignore_index=True)
print(summary.to_string(index=False))
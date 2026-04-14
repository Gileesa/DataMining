# Comparison
import pandas as pd

# Using quantile - Random Forest 50 vs previous approvh 87%
# Now we dont have NAN values for XGBoost
rf_results = pd.read_csv("DataMining/csv_files/rf_results.csv")
xgb_results = pd.read_csv("DataMining/csv_files/xgb_results.csv")

summary = pd.concat([rf_results, xgb_results], ignore_index=True)
print(summary.to_string(index=False))
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data

# Create folders
os.makedirs("DataMining/Figures/Regression/RandomForest", exist_ok=True)

# Load original data again
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Load predictions if you already saved them
pred_path = "DataMining/csv_files/Regression/RandomForest/rf_tuned_predictions.csv"

if os.path.exists(pred_path):
    results_df = pd.read_csv(pred_path)
else:
    results_df = None
    print("Prediction file not found:", pred_path)

# 1. Correlation heatmap
corr_df = train_df[feature_cols + ["target"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_correlation_matrix.png", dpi=150)
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_correlation_matrix.pdf")
plt.close()

# 2. Target distribution
plt.figure(figsize=(7, 5))
train_df["target"].hist(bins=30)
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_target_distribution.png", dpi=150)
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_target_distribution.pdf")
plt.close()

# 3. Actual vs Predicted plot
if results_df is not None:
    plt.figure(figsize=(7, 5))
    plt.scatter(results_df["actual_value"], results_df["predicted_value"], alpha=0.6)
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("DataMining/Figures/Regression/RandomForest/rf_actual_vs_predicted.png", dpi=150)
    plt.savefig("DataMining/Figures/Regression/RandomForest/rf_actual_vs_predicted.pdf")
    plt.close()

    # 4. Residual plot
    results_df["residual"] = results_df["actual_value"] - results_df["predicted_value"]

    plt.figure(figsize=(7, 5))
    plt.scatter(results_df["predicted_value"], results_df["residual"], alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("DataMining/Figures/Regression/RandomForest/rf_residual_plot.png", dpi=150)
    plt.savefig("DataMining/Figures/Regression/RandomForest/rf_residual_plot.pdf")
    plt.close()

print("Plots saved successfully.")
import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder for saving figures
fig_dir = "DataMining/Figures/Regression/TEST"

# Folder with baseline regression results
baseline_csv_dir = "DataMining/csv_files/Regression/RandomForest"

# Folder with top-20 regression results
improved_csv_dir = "DataMining/csv_files/Regression/TEST"

# Create figure folder if it does not exist
os.makedirs(fig_dir, exist_ok=True)

# File paths for result tables
baseline_results_path = baseline_csv_dir + "/rf_tuned_results.csv"
improved_results_path = improved_csv_dir + "/rf_top20_results.csv"

# File paths for prediction tables
baseline_pred_path = baseline_csv_dir + "/rf_tuned_predictions.csv"
improved_pred_path = improved_csv_dir + "/rf_top20_predictions.csv"

# File path for feature importance table
importance_path = baseline_csv_dir + "/rf_feature_importances_sorted.csv"

# Load csv files
baseline_results_df = pd.read_csv(baseline_results_path)
improved_results_df = pd.read_csv(improved_results_path)
baseline_pred_df = pd.read_csv(baseline_pred_path)
improved_pred_df = pd.read_csv(improved_pred_path)
feat_imp_df = pd.read_csv(importance_path)

# Create a dataframe for model comparison
comparison_df = pd.DataFrame()


# Add model names
model_names = []
model_names.append("Baseline Random Forest Regressor")
model_names.append("Top 20 Random Forest Regressor")
comparison_df["Model"] = model_names

# Add CV MSE scores
cv_mse_scores = []
cv_mse_scores.append(baseline_results_df.loc[0, "Best CV MSE"])
cv_mse_scores.append(improved_results_df.loc[0, "Best CV MSE"])
comparison_df["CV MSE"] = cv_mse_scores

# Add test MSE scores
mse_scores = []
mse_scores.append(baseline_results_df.loc[0, "Test MSE"])
mse_scores.append(improved_results_df.loc[0, "Test MSE"])
comparison_df["Test MSE"] = mse_scores

# Add test MAE scores
mae_scores = []
mae_scores.append(baseline_results_df.loc[0, "Test MAE"])
mae_scores.append(improved_results_df.loc[0, "Test MAE"])
comparison_df["Test MAE"] = mae_scores

# Calculate percentage improvement from baseline to top 20
baseline_cv_mse = comparison_df.loc[0, "CV MSE"]
top20_cv_mse = comparison_df.loc[1, "CV MSE"]

baseline_test_mse = comparison_df.loc[0, "Test MSE"]
top20_test_mse = comparison_df.loc[1, "Test MSE"]

baseline_test_mae = comparison_df.loc[0, "Test MAE"]
top20_test_mae = comparison_df.loc[1, "Test MAE"]

cv_mse_improvement = ((baseline_cv_mse - top20_cv_mse) / baseline_cv_mse) * 100
test_mse_improvement = ((baseline_test_mse - top20_test_mse) / baseline_test_mse) * 100
test_mae_improvement = ((baseline_test_mae - top20_test_mae) / baseline_test_mae) * 100

# Save comparison table
# Save comparison table
comparison_df.to_csv(improved_csv_dir + "/rf_model_comparison_table.csv", index=False)

# Prepare dataframe for plotting
comparison_plot = comparison_df.set_index("Model")[["CV MSE", "Test MSE", "Test MAE"]]

ax = comparison_plot.plot(
    kind="bar",
    figsize=(8, 6),
    width=0.45
)

plt.title("Random Forest Regression: Baseline vs Top 20 Features", fontsize=16)
plt.xlabel("Model", fontsize=13)
plt.ylabel("Error", fontsize=13)
plt.xticks(rotation=0, ha="center")
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Add more headroom for labels
plt.ylim(0, 0.50)

for container in ax.containers:
    labels = []
    for value in container.datavalues:
        labels.append(f"{value:.3f}")
    ax.bar_label(container, labels=labels, padding=3, fontsize=10)

# Move legend outside the plot
plt.legend(
    title="Metric",
    fontsize=11,
    title_fontsize=11,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0)
)

plt.tight_layout()
plt.savefig(f"{fig_dir}/rf_model_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{fig_dir}/rf_model_comparison.pdf", bbox_inches="tight")
plt.close()


# Create residuals for baseline model
baseline_pred_df["residual"] = baseline_pred_df["actual_value"] - baseline_pred_df["predicted_value"]

# Plot residuals for baseline model
plt.figure(figsize=(7, 5))
plt.scatter(baseline_pred_df["predicted_value"], baseline_pred_df["residual"], alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Baseline Random Forest Residual Plot")
plt.tight_layout()
plt.savefig(fig_dir + "/baseline_residual_plot.png", dpi=150)
plt.savefig(fig_dir + "/baseline_residual_plot.pdf")
plt.close()

# Create residuals for top-20 model
improved_pred_df["residual"] = improved_pred_df["actual_value"] - improved_pred_df["predicted_value"]

# Plot residuals for top 20 model
plt.figure(figsize=(7, 5))
plt.scatter(improved_pred_df["predicted_value"], improved_pred_df["residual"], alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Top 20 Random Forest Residual Plot")
plt.tight_layout()
plt.savefig(fig_dir + "/top20_residual_plot.png", dpi=150)
plt.savefig(fig_dir + "/top20_residual_plot.pdf")
plt.close()

# Select the 20 most important features
top_k = 20
top_features_df = feat_imp_df.head(top_k)

# Sort them for a cleaner bar plot
plot_df = top_features_df.sort_values(by="importance", ascending=True)

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest Regression Features")
plt.tight_layout()
plt.savefig(fig_dir + "/rf_top20_feature_importance.png", dpi=150)
plt.savefig(fig_dir + "/rf_top20_feature_importance.pdf")
plt.close()

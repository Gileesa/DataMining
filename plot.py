import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data

fig_dir = "DataMining/Figures/Classification/TEST"
csv_dir = "DataMining/csv_files/Classification/TEST"

os.makedirs(fig_dir, exist_ok=True)

# Load train/test data
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Load saved CSV files
comparison_path = f"{csv_dir}/rf_feature_selection_cv_results.csv"
importance_path = f"{csv_dir}/rf_feature_importances_sorted.csv"
top20_path = f"{csv_dir}/top20_selected_features.csv"
final_pred_path = f"{csv_dir}/rf_final_predictions.csv"
final_results_path = f"{csv_dir}/rf_final_results.csv"

comparison_df = pd.read_csv(comparison_path)
feat_imp_df = pd.read_csv(importance_path)
top20_df = pd.read_csv(top20_path)
final_results_df = pd.read_csv(final_pred_path)
summary_df = pd.read_csv(final_results_path)

# Class distribution
plt.figure(figsize=(7, 5))
train_df["target_classification"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{fig_dir}/class_distribution.png", dpi=150)
plt.savefig(f"{fig_dir}/class_distribution.pdf")
plt.close()

# Correlation matrix
corr_df = train_df[feature_cols + ["target"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{fig_dir}/rf_correlation_matrix.png", dpi=150)
plt.savefig(f"{fig_dir}/rf_correlation_matrix.pdf")
plt.close()

# Final model confusion matrix
final_cm = pd.crosstab(
    final_results_df["actual_class"],
    final_results_df["predicted_class"]
)

selected_model_name = summary_df.loc[0, "Selected Model"]

plt.figure(figsize=(6, 5))
sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{selected_model_name} - Final Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(f"{fig_dir}/rf_final_confusion_matrix.png", dpi=150)
plt.savefig(f"{fig_dir}/rf_final_confusion_matrix.pdf")
plt.close()

# Comparison bar chart
selected_model_name = summary_df.loc[0, "Selected Model"]
final_test_accuracy = summary_df.loc[0, "Test Accuracy"]
final_test_macro_f1 = summary_df.loc[0, "Test Macro F1"]

plot_df = pd.DataFrame()
plot_df["Model"] = comparison_df["Model"]
plot_df["CV macro F1 (train)"] = comparison_df["Cross-validation Macro F1"]
plot_df["Test Accuracy"] = 0.0
plot_df["Test Macro F1"] = 0.0

if selected_model_name == "Random Forest (all features)":
    plot_df.loc[0, "Test Accuracy"] = final_test_accuracy
    plot_df.loc[0, "Test Macro F1"] = final_test_macro_f1
else:
    plot_df.loc[1, "Test Accuracy"] = final_test_accuracy
    plot_df.loc[1, "Test Macro F1"] = final_test_macro_f1

comparison_plot = plot_df.set_index("Model")[["CV macro F1 (train)", "Test Accuracy", "Test Macro F1"]]

# scale to percentages
comparison_plot_percent = comparison_plot * 100

ax = comparison_plot_percent.plot(
    kind="bar",
    figsize=(8, 6),
    width=0.75
)

plt.title("Random Forest Performance: All Features vs Top 20 Features", fontsize=16)
plt.xlabel("Model", fontsize=13)
plt.ylabel("Score (%)", fontsize=13)
plt.ylim(0, 100)
plt.xticks(rotation=0, ha="center")
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for container in ax.containers:
    labels = []
    for value in container.datavalues:
        if value > 0:
            labels.append(f"{value:.1f}")
        else:
            labels.append("")
    ax.bar_label(container, labels=labels, padding=3, fontsize=10)

plt.legend(title="Metric", fontsize=11, title_fontsize=11)
plt.tight_layout()
plt.savefig(f"{fig_dir}/rf_feature_selection_comparison.png", dpi=150)
plt.savefig(f"{fig_dir}/rf_feature_selection_comparison.pdf")
plt.close()

# Top-20 feature importance plot
top_k = 20
top_features_df = feat_imp_df.head(top_k)
plot_df = top_features_df.sort_values("importance", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest Features")
plt.tight_layout()
plt.savefig(f"{fig_dir}/rf_top20_feature_importance.png", dpi=150)
plt.savefig(f"{fig_dir}/rf_top20_feature_importance.pdf")
plt.close()

# Histograms of selected top features
selected_features = top20_df["selected_feature"].tolist()

selected_existing = []
for feature in selected_features:
    if feature in train_df.columns:
        selected_existing.append(feature)

hist_features = []
max_features_for_hist = 6

counter = 0
for feature in selected_existing:
    if counter < max_features_for_hist:
        hist_features.append(feature)
        counter = counter + 1

ncols = 2
nrows = math.ceil(len(hist_features) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))

if isinstance(axes, plt.Axes):
    axes = [axes]
else:
    axes = axes.flatten()

i = 0
while i < len(hist_features):
    feature_name = hist_features[i]
    ax = axes[i]

    values = train_df[feature_name].dropna()

    ax.hist(values, bins=20, edgecolor="black")
    ax.set_title(feature_name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    i = i + 1

j = len(hist_features)
while j < len(axes):
    axes[j].set_visible(False)
    j = j + 1

fig.suptitle("Histograms of Selected Top Features", fontsize=14)
plt.tight_layout()
plt.savefig(f"{fig_dir}/selected_feature_histograms.png", dpi=150)
plt.savefig(f"{fig_dir}/selected_feature_histograms.pdf")
plt.close()

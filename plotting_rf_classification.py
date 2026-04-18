import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


fig_dir = "DataMining/Figures/Classification/RandomForest"
csv_dir = "DataMining/csv_files/Classification/RandomForest"

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)


# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Load predictions
pred_path = f"{csv_dir}/rf_predictions.csv"

if os.path.exists(pred_path):
    results_df = pd.read_csv(pred_path)
else:
    results_df = None
    print(f"Prediction file not found: {pred_path}")


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


# Actual vs Predicted class counts

if results_df is not None:
    compare_df = pd.crosstab(
        results_df["actual_class"],
        results_df["predicted_class"]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(compare_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Actual vs Predicted Classes")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/actual_vs_predicted_classes.png", dpi=150)
    plt.savefig(f"{fig_dir}/actual_vs_predicted_classes.pdf")
    plt.close()

print("Classification plots saved successfully.")
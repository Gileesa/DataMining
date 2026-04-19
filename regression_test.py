from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import os

from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/Figures/Regression/TEST", exist_ok=True)
os.makedirs("DataMining/csv_files/Regression/TEST", exist_ok=True)

# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

y_train = train_df["target"]
y_test = test_df["target"]

# Load top 20  features from baseline model
top20_df = pd.read_csv(
    "DataMining/csv_files/Regression/RandomForest/rf_feature_importances_sorted.csv"
)

top_k = 20
top20_df = top20_df.head(top_k)

selected_feature_names = top20_df["feature"].tolist()

print("\nSelected top 20 features from baseline model:")
print(selected_feature_names)

selected_feature_df = pd.DataFrame()
selected_feature_df["selected_feature"] = selected_feature_names
selected_feature_df.to_csv(
    "DataMining/csv_files/Regression/TEST/top20_selected_features.csv",
    index=False
)

# Get indices of selected features
selected_indices = pd.Index(feature_cols).get_indexer(selected_feature_names)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Hyperparameter space
rf_param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 3, 5, 7],
    "max_features": ["sqrt", "log2", 0.5, 1.0]
}

# Train Random Forest on top 20 features
rf_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=rf_param_grid,
    n_iter=100,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    refit=True,
    cv=tscv,
    verbose=1,
    pre_dispatch="2*n_jobs",
    random_state=RANDOM_STATE,
    error_score="raise",
    return_train_score=False
)

rf_search.fit(X_train_selected, y_train)

print("\nTOP 20 RANDOM SEARCH RESULTS")
print("Best parameters:")
for k, v in rf_search.best_params_.items():
    print(f"{k}: {v}")

print(f"Best CV score (neg MSE): {rf_search.best_score_:.4f}")
print(f"Best CV MSE: {-rf_search.best_score_:.4f}")
print("Best estimator:")
print(rf_search.best_estimator_)

best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_selected)

test_mse = mean_squared_error(y_test, y_pred_rf)
test_mae = mean_absolute_error(y_test, y_pred_rf)

print("\nTOP 20 TEST RESULTS")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")

# Save predictions
results_df = pd.DataFrame()
results_df["id"] = test_df["id"].values
results_df["date"] = test_df["date"].values
results_df["actual_value"] = y_test.values
results_df["predicted_value"] = y_pred_rf

results_df.to_csv(
    "DataMining/csv_files/Regression/TEST/rf_top20_predictions.csv",
    index=False
)

# Save summary
summary_df = pd.DataFrame({
    "Model": ["Random Forest Regressor (top 20 features)"],
    "Test MSE": [test_mse],
    "Test MAE": [test_mae],
    "Best Params": [str(rf_search.best_params_)],
    "Best CV Neg MSE": [rf_search.best_score_],
    "Best CV MSE": [-rf_search.best_score_],
})

summary_df.to_csv(
    "DataMining/csv_files/Regression/TEST/rf_top20_results.csv",
    index=False
)

# Save top 20 feature importance plot from baseline ranking
plot_df = top20_df.sort_values(by="importance", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest Regression Features")
plt.tight_layout()
plt.savefig("DataMining/Figures/Regression/TEST/rf_top20_feature_importance.png", dpi=150)
plt.savefig("DataMining/Figures/Regression/TEST/rf_top20_feature_importance.pdf")
plt.close()
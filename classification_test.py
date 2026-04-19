from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

import pandas as pd
import os

from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/csv_files/Classification/TEST", exist_ok=True)

# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df["target_classification"])
y_test = le.transform(test_df["target_classification"])

print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Hyperparameter space
rf_param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [2, 4, 6, None],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 3, 5, 7, 9],
    "max_features": [None, "sqrt", "log2"]
}

# Train full-feature model
rf_search_full = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    param_distributions=rf_param_grid,
    n_iter=100,
    cv=tscv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

rf_search_full.fit(X_train, y_train)
best_rf_full = rf_search_full.best_estimator_

print("\nOriginal model with all features")
print(f"Best params: {rf_search_full.best_params_}")
print(f"Best CV macro F1: {rf_search_full.best_score_:.4f}")

# Get feature importances
if hasattr(X_train, "columns"):
    feature_names = X_train.columns.tolist()
else:
    feature_names = feature_cols

feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": best_rf_full.feature_importances_
})

feat_imp_df = feat_imp_df.sort_values(by="importance", ascending=False)

print("\nTop 20 important features:")
print(feat_imp_df.head(20))

feat_imp_df.to_csv(
    "DataMining/csv_files/Classification/TEST/rf_feature_importances_sorted.csv",
    index=False
)

# Select top 20 features
top_k = 20
selected_feature_names = feat_imp_df.head(top_k)["feature"].tolist()

print(f"\nSelected top {top_k} features:")
print(selected_feature_names)

selected_feature_df = pd.DataFrame()
selected_feature_df["selected_feature"] = selected_feature_names
selected_feature_df.to_csv(
    "DataMining/csv_files/Classification/TEST/top20_selected_features.csv",
    index=False
)

# Get indices for selected features
selected_indices = pd.Index(feature_cols).get_indexer(selected_feature_names)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Train model on selected features
rf_search_selected = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    param_distributions=rf_param_grid,
    n_iter=100,
    cv=tscv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

rf_search_selected.fit(X_train_selected, y_train)
best_rf_selected = rf_search_selected.best_estimator_

print("\nReduced model (top 20 features)")
print(f"Best params: {rf_search_selected.best_params_}")
print(f"Best CV macro F1: {rf_search_selected.best_score_:.4f}")

# Compare models using cross-validation
comparison_df = pd.DataFrame({
    "Model": ["Random Forest (all features)", "Random Forest (top 20 features)"],
    "Cross-validation Macro F1": [rf_search_full.best_score_, rf_search_selected.best_score_]
})

print("\nComparison based on cross-validation:")
print(comparison_df)

comparison_df.to_csv(
    "DataMining/csv_files/Classification/TEST/rf_feature_selection_cv_results.csv",
    index=False
)

# Choose final model based on CV performance
if rf_search_selected.best_score_ >= rf_search_full.best_score_:
    final_model = best_rf_selected
    X_test_final = X_test_selected
    final_model_name = "Random Forest (top 20 features)"
    final_cv_macro_f1 = rf_search_selected.best_score_
else:
    final_model = best_rf_full
    X_test_final = X_test
    final_model_name = "Random Forest (all features)"
    final_cv_macro_f1 = rf_search_full.best_score_

print("\nFinal selected model based on cross-validation:")
print(final_model_name)

# Evaluate final model on test set once
y_pred_final = final_model.predict(X_test_final)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_macro_f1 = f1_score(y_test, y_pred_final, average="macro")

print("\nFinal test results")
print(f"Selected model: {final_model_name}")
print(f"Test Accuracy : {final_accuracy:.4f}")
print(f"Test Macro F1 : {final_macro_f1:.4f}")

# Save final predictions
actual_class_final = le.inverse_transform(y_test)
predicted_class_final = le.inverse_transform(y_pred_final)

final_results_df = pd.DataFrame()
final_results_df["id"] = test_df["id"].values
final_results_df["date"] = test_df["date"].values
final_results_df["actual_value"] = test_df["target"].values
final_results_df["actual_class"] = actual_class_final
final_results_df["predicted_class"] = predicted_class_final

final_results_df.to_csv(
    "DataMining/csv_files/Classification/TEST/rf_final_predictions.csv",
    index=False
)

# Save classification report
report_final = classification_report(
    y_test,
    y_pred_final,
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

report_final_df = pd.DataFrame(report_final).transpose()
report_final_df.to_csv(
    "DataMining/csv_files/Classification/TEST/rf_final_classification_report.csv"
)

# Save summary
summary_df = pd.DataFrame({
    "Selected Model": [final_model_name],
    "Cross-validation Macro F1": [final_cv_macro_f1],
    "Test Accuracy": [final_accuracy],
    "Test Macro F1": [final_macro_f1]
})

summary_df.to_csv(
    "DataMining/csv_files/Classification/TEST/rf_final_results.csv",
    index=False
)
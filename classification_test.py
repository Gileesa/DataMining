from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import os

from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/csv_files/Classification/TEST", exist_ok=True)

# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['target_classification'])
y_test = le.transform(test_df['target_classification'])

print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Time-aware CV
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Hyperparameter space
rf_param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "criterion": ['gini', 'entropy', 'log_loss'],
    "max_depth": [2, 4, 6, None],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 3, 5, 7, 9],
    "max_features": [None, 'sqrt', 'log2']
}

#Train full-feature model
rf_search_full = RandomizedSearchCV(
                                    estimator   = RandomForestClassifier(
                                    random_state= RANDOM_STATE,
                                    class_weight= 'balanced'
                                     ),
                                    param_distributions=rf_param_grid,
                                    n_iter=100,
                                    cv=tscv,
                                    scoring='f1_macro',
                                    n_jobs=-1,
                                    random_state=RANDOM_STATE,
                                    verbose=1
                                )

rf_search_full.fit(X_train, y_train)
best_rf_full = rf_search_full.best_estimator_

y_pred_full = best_rf_full.predict(X_test)
full_accuracy = accuracy_score(y_test, y_pred_full)
full_macro_f1 = f1_score(y_test, y_pred_full, average='macro')

print("\n Original model with all features")
print(f"Best params: {rf_search_full.best_params_}")
print(f"Best CV macro F1: {rf_search_full.best_score_:.4f}")
print(f"Test Accuracy : {full_accuracy:.4f}")
print(f"Test Macro F1 : {full_macro_f1:.4f}")

# Get feature importances
if hasattr(X_train, 'columns'):
    feature_names = X_train.columns.tolist()
else:
    feature_names = feature_cols

feat_imp_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': best_rf_full.feature_importances_
                        }).sort_values(by='importance', ascending=False)

print("\nTop 20 important features:")
print(feat_imp_df.head(20))

# Select top-k features
top_k = 20
selected_feature_names = feat_imp_df.head(top_k)["feature"].tolist()

print(f"\nSelected top {top_k} features:")
print(selected_feature_names)

# Get positions of selected features in feature_cols
selected_indices = pd.Index(feature_cols).get_indexer(selected_feature_names)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# 4. Retrain on selected features
rf_search_selected = RandomizedSearchCV(
                                        estimator=RandomForestClassifier(
                                        random_state=RANDOM_STATE,
                                        class_weight='balanced'
                                        
                                        ),
                                        param_distributions=rf_param_grid,
                                        n_iter=100,
                                        cv=tscv,
                                        scoring='f1_macro',
                                        n_jobs=-1,
                                        random_state=RANDOM_STATE,
                                        verbose=1
)

rf_search_selected.fit(X_train_selected, y_train)
best_rf_selected = rf_search_selected.best_estimator_

y_pred_selected = best_rf_selected.predict(X_test_selected)
selected_accuracy = accuracy_score(y_test, y_pred_selected)
selected_macro_f1 = f1_score(y_test, y_pred_selected, average='macro')

print("\n Reduced model (top 20 features)")
print(f"Best params: {rf_search_selected.best_params_}")
print(f"Best CV macro F1: {rf_search_selected.best_score_:.4f}")
print(f"Test Accuracy : {selected_accuracy:.4f}")
print(f"Test Macro F1 : {selected_macro_f1:.4f}")

# Compare results
comparison_df = pd.DataFrame({
    'Model': ['Random Forest (all features)', 'Random Forest (top 20 features)'],
    'CV macro F1 (train)': [rf_search_full.best_score_, rf_search_selected.best_score_],
    'Test Accuracy': [full_accuracy, selected_accuracy],
    'Test Macro F1': [full_macro_f1, selected_macro_f1]
})

print("\nComparison:")
print(comparison_df)

comparison_df.to_csv(
    'DataMining/csv_files/Classification/TEST/rf_feature_selection_trial_results.csv',
    index=False
)

feat_imp_df.to_csv(
    'DataMining/csv_files/Classification/TEST/rf_feature_importances_sorted.csv',
    index=False
)


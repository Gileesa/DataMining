from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/Figures/Classification/GridSeach", exist_ok=True)
os.makedirs("DataMining/csv_files/Classification/GridSeach", exist_ok=True)

# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['target_classification'])
y_test = le.transform(test_df['target_classification'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel mapping:", label_mapping)

# Time-aware CV
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Smaller grid first, otherwise GridSearch can become very slow
rf_param_grid = {
    "n_estimators"      : [100, 200, 300, 400, 500],
    "criterion"         : ['gini', "entropy", "log_loss"], 
    "max_depth"         : [2, 4, 6, None], 
    "min_samples_split" : [2, 4, 6, 8, 10], 
    "min_samples_leaf"  : [1, 3, 5, 7, 9],
    "max_features"      : [None, 'sqrt', 'log2'],
}

# Grid Search
rf_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True
)

# Train
rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV macro F1: {rf_search.best_score_:.4f}")

# Test evaluation
y_pred_rf = best_rf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_rf)
test_macro_f1 = f1_score(y_test, y_pred_rf, average='macro')

print("\n── Test set results ──")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Macro F1 : {test_macro_f1:.4f}")

print("\nClassification report:")
label_ids = le.transform(le.classes_)
print(classification_report(
    y_test,
    y_pred_rf,
    labels=label_ids,
    target_names=le.classes_,
    zero_division=0
))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=label_ids)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/GridSeach/rf_confusion_matrix.png", dpi=150)
plt.savefig("DataMining/Figures/Classification/GridSeach/rf_confusion_matrix.pdf")
plt.close()

# Feature importance
feat_imp = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values()
feat_imp.plot(kind='barh', figsize=(7, 6), title="RF Feature Importances")
plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/GridSeach/rf_feature_importance.pdf")
plt.savefig("DataMining/Figures/Classification/GridSeach/rf_feature_importance.png", dpi=150)
plt.close()

# Save predictions
predicted_labels = le.inverse_transform(y_pred_rf)
actual_labels = le.inverse_transform(y_test)

results_df = pd.DataFrame({
    'id': test_df['id'].values,
    'date': test_df['date'].values,
    'actual_value': test_df['target'].values,
    'actual_class': actual_labels,
    'predicted_class': predicted_labels
})

print("\nPredictions for each id and date:")
print(results_df.head(20))

results_df.to_csv("DataMining/csv_files/Classification/GridSeach/rf_predictions.csv", index=False)

rf_results = pd.DataFrame({
    'Model': ['Random Forest'],
    'CV macro F1 (train)': [rf_search.best_score_],
    'Test Accuracy': [test_accuracy],
    'Test macro F1': [test_macro_f1]
})
rf_results.to_csv("DataMining/csv_files/Classification/GridSeach/rf_results.csv", index=False)
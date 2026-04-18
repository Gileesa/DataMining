from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import N_SPLITS, RANDOM_STATE
from data_loader import load_data
from sklearn.utils.class_weight import compute_sample_weight

# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

os.makedirs("DataMining/Figures/Classification/XGBOOST", exist_ok=True)
os.makedirs("DataMining/csv_files/Classification/XGBOOST", exist_ok=True)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['target_classification'])
y_test  = le.transform(test_df['target_classification'])

label_mapping = {}
labels = le.classes_
numbers = le.transform(le.classes_)

for i in range(len(labels)):
    label = labels[i]
    number = numbers[i]
    label_mapping[label] = number
print("\nLabel mapping:", label_mapping)

# TimeSeriesSplit for cross-validation - time-aware splitting
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# MODEL 2 - XG Boost
xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],

    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],

    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5, 1],

    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 2, 5, 10],
}

# Search for the best model
xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        tree_method='hist'
    ),
    param_distributions=xgb_param_grid,
    n_iter=100,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
    error_score='raise'
)

# Train the model on the training set

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
xgb_search.fit(X_train, y_train, sample_weight=sample_weights)

# Best XGB model from search
best_xgb = xgb_search.best_estimator_
print(f"\nBest XGB params: {xgb_search.best_params_}")
print(f"Best CV macro F1: {xgb_search.best_score_:.4f}")

# Evaluate on the test set
y_pred_xgb = best_xgb.predict(X_test)
test_f1_score = f1_score(y_test, y_pred_xgb, average='macro')
test_accuracy_score = accuracy_score(y_test, y_pred_xgb)

print("\n── Test set results ──")
print(f"Accuracy : {test_accuracy_score:.4f}")
print(f"Macro F1 : {test_f1_score:.4f}")

print("Best params:", xgb_search.best_params_)
print("Best score:", xgb_search.best_score_)
# print("Mean test scores:", xgb_search.cv_results_['mean_test_score'])

print("\nClassification report:")
all_labels = le.transform(le.classes_)
print(classification_report(
                            y_test, 
                            y_pred_xgb, 
                            labels=all_labels,
                            target_names=le.classes_,
                            zero_division=0
                            ))

# Confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels = all_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("XGBoost – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/XGBOOST/xgb_confusion_matrix.png", dpi=150)
plt.savefig("DataMining/Figures/Classification/XGBOOST/xgb_confusion_matrix.pdf")
plt.close()

# Feature importance plot
feat_imp = pd.Series(best_xgb.feature_importances_, index=feature_cols)

feat_imp = feat_imp.sort_values()
feat_imp.plot(kind='barh', figsize=(7, 6),title="XGB Feature Importances")

plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/XGBOOST/xgb_feature_importance.pdf")
plt.savefig("DataMining/Figures/Classification/XGBOOST/xgb_feature_importance.png", dpi = 150)
plt.close()

# Convert predicted numbers back to class names
predicted_labels = le.inverse_transform(y_pred_xgb)
actual_labels = le.inverse_transform(y_test)

# Building a table with id, date, actual class, predicted class
results_df = pd.DataFrame()
results_df['id'] = test_df['id'].values
results_df['date'] = test_df['date'].values
results_df['actual_value'] = test_df['target'].values
results_df['actual_class'] = actual_labels
results_df['predicted_class'] = predicted_labels

print("\nPredictions for each id and date:")
print(results_df.head(20))

# Save to CSV
results_df.to_csv("DataMining/csv_files/Classification/XGBOOST/xgb_predictions.csv", index=False)

xgb_results = pd.DataFrame({
    'Model': ['XGBoost'],
    'CV macro F1 (train)': [xgb_search.best_score_],
    'Test Accuracy': [test_accuracy_score],
    'Test macro F1': [test_f1_score]
})

xgb_results.to_csv("DataMining/csv_files/Classification/XGBOOST/xgb_results.csv", index=False)
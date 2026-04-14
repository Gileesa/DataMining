from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_data

# Issue few people with a low mood !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Configuration
N_SPLITS   = 5 
RANDOM_STATE = 42

# Load data
train_df, test_df, X_train, X_test, FEATURE_COLS = load_data()

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

# MODEL 1 — Random Forest

rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 5, 10, 15],
    'min_samples_leaf':  [1, 2, 5],
    'max_features':      ['sqrt', 'log2'],
}

# Search for the best model
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=tscv,
    scoring='f1_macro',      # macro F1 — handles class imbalance
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

# Train the model on the training set
rf_search.fit(X_train, y_train)

# Best RF model from search
best_rf = rf_search.best_estimator_
print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV macro F1: {rf_search.best_score_:.4f}")

# Evaluate on the test set
y_pred_rf = best_rf.predict(X_test)
print("\n── Test set results ──")
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_rf, average='macro'):.4f}")

print("\nClassification report:")
label_names = le.transform(le.classes_)
print(classification_report(
                            y_test, 
                            y_pred_rf, 
                            labels=label_names,
                            target_names=le.classes_,
                            zero_division=0
                            ))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf, labels = label_names)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/figures/rf_confusion_matrix.png", dpi=150)
plt.savefig("DataMining/figures/rf_confusion_matrix.pdf")
#plt.show()

# Feature importance plot
feat_imp = pd.Series(best_rf.feature_importances_, index=FEATURE_COLS)

feat_imp = feat_imp.sort_values()
feat_imp.plot(kind='barh', figsize=(7, 6),title="RF Feature Importances")

plt.tight_layout()
plt.savefig("DataMining/figures/rf_feature_importance.pdf")
plt.savefig("DataMining/figures/rf_feature_importance.png", dpi = 150)
# plt.show()

# Convert predicted numbers back to class names
predicted_labels = le.inverse_transform(y_pred_rf)
actual_labels = le.inverse_transform(y_test)

# Building a table with id, date, actual class, predicted class
results_df = pd.DataFrame()
results_df['id'] = test_df['id'].values
results_df['date'] = test_df['date'].values
results_df['actual_class'] = actual_labels
results_df['predicted_class'] = predicted_labels

print("\nPredictions for each id and date:")
print(results_df.head(20))

# Save to CSV
results_df.to_csv("DataMining/csv_files/rf_predictions.csv", index=False)
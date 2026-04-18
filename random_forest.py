from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/Figures/Classification/RandomForest", exist_ok=True)
os.makedirs("DataMining/csv_files/Classification/RandomForest", exist_ok=True)


# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

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

    "n_estimators"      : [100, 200, 300, 400, 500],
    "criterion"         : ['gini', "entropy", "log_loss"], 
    "max_depth"         : [2, 4, 6, None], 
    "min_samples_split" : [2, 4, 6, 8, 10], 
    "min_samples_leaf"  : [1, 3, 5, 7, 9],
    # "min_weight_fraction_leaf"  : [0.0], 
    "max_features"      : [None, 'sqrt', 'log2'], 
    # "max_leaf_nodes"    : [None], 
    # "min_impurity_decrease" : [0.0], 
    # "bootstrap"         : [True], 
    # "oob_score"         : [False], 
    # "n_jobs"            : [None], 
    # "random_state"      : [None], 
    # "verbose"           : [0], 
    # "warm_start"        : [False],
    # "class_weight"      : [None], 
    # "ccp_alpha"         : [0.0], 
    # "max_samples"       : [None],
    # "monotonic_cst"     : [None]
}

# Search for the best model
rf_search = RandomizedSearchCV(
    RandomForestClassifier(
                            random_state        =RANDOM_STATE, 
                            class_weight        ='balanced'),
                            param_distributions =rf_param_grid,
                            n_iter              =100,
                            cv                  =tscv,
                            scoring             ='f1_macro',      # macro F1 — handles class imbalance
                            n_jobs              =-1,
                            random_state        =RANDOM_STATE,
                            verbose             =1,
                        )

# Train the model on the training set
rf_search.fit(X_train, y_train)

# Best RF model from search
best_rf = rf_search.best_estimator_


print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV macro F1: {rf_search.best_score_:.4f}")

# Evaluate on the test set
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
cm_rf = confusion_matrix(y_test, y_pred_rf, labels = label_ids)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/RandomForest/rf_confusion_matrix.png", dpi=150)
plt.savefig("DataMining/Figures/Classification/RandomForest/rf_confusion_matrix.pdf")

#plt.show()
plt.close()

# Feature importance plot
feat_imp = pd.Series(best_rf.feature_importances_, index=feature_cols)

feat_imp = feat_imp.sort_values()
feat_imp.plot(kind='barh', figsize=(7, 6),title="RF Feature Importances")

plt.tight_layout()
plt.savefig("DataMining/Figures/Classification/RandomForest/rf_feature_importance.pdf")
plt.savefig("DataMining/Figures/Classification/RandomForest/rf_feature_importance.png", dpi = 150)
# plt.show()
plt.close()


# Convert predicted numbers back to class names
predicted_labels = le.inverse_transform(y_pred_rf)
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
results_df.to_csv("DataMining/csv_files/Classification/RandomForest/rf_predictions.csv", index=False)

rf_results = pd.DataFrame({
    'Model': ['Random Forest'],
    'CV macro F1 (train)': [rf_search.best_score_],
    'Test Accuracy': [test_accuracy],
    'Test macro F1': [test_macro_f1]
})

rf_results.to_csv("DataMining/csv_files/Classification/RandomForest/rf_results.csv", index=False)
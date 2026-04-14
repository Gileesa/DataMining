import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configuration
DATE_COL   = 'date'
META_COLS = ['id', 'date', 'target', 'target_classification']

# Data loading
def load_data(train_path="DataMining/csv_files/mood_train.csv", test_path="DataMining/csv_files/mood_test.csv"):
    """
    Load and preprocess the training and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])

    print(train_df.head())
    print(f"\nTrain class distribution:\n{train_df['target_classification'].value_counts()}")
    print(f"\nTest class distribution:\n{test_df['target_classification'].value_counts()}")
    print(f"\nTrain: {len(train_df)} rows | Test: {len(test_df)} rows")

    feature_col = []

    for col in train_df.columns:
        if col not in META_COLS:
            feature_col.append(col)

    X_train = train_df[feature_col].values
    X_test  = test_df[feature_col].values

    return train_df, test_df, X_train, X_test, feature_col
















# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['target_classification'])
y_test  = le.transform(test_df['target_classification'])
print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


# MODEL 1 — Random Forest
print("\n" + "="*60)
print("MODEL 1: Random Forest")
print("="*60)

rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 5, 10, 15],
    'min_samples_leaf':  [1, 2, 5],
    'max_features':      ['sqrt', 'log2'],
}

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
rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV macro F1: {rf_search.best_score_:.4f}")

# Evaluate on held-out test set
y_pred_rf = best_rf.predict(X_test)
print("\n── Test set results ──")
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/figures/rf_confusion_matrix.png", dpi=150)
plt.show()

# Feature importance plot
feat_imp = pd.Series(best_rf.feature_importances_, index=FEATURE_COLS)
feat_imp.sort_values().plot(kind='barh', figsize=(7, 6),
                             title="RF Feature Importances")
plt.tight_layout()
plt.savefig("DataMining/figures/rf_feature_importance.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2: XGBoost")
print("="*60)

# XGBoost needs integer labels starting from 0 — LabelEncoder already gives that
xgb_param_grid = {
    'n_estimators':   [100, 200, 300],
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.05, 0.1, 0.2],
    'subsample':      [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    ),
    param_distributions=xgb_param_grid,
    n_iter=20,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)
xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
print(f"\nBest XGB params: {xgb_search.best_params_}")
print(f"Best CV macro F1: {xgb_search.best_score_:.4f}")

# Evaluate on held-out test set
y_pred_xgb = best_xgb.predict(X_test)
print("\n── Test set results ──")
print(f"Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("XGBoost – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DataMining/figures/xgb_confusion_matrix.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
summary = pd.DataFrame({
    'Model':    ['Random Forest', 'XGBoost'],
    'CV macro F1 (train)': [rf_search.best_score_,  xgb_search.best_score_],
    'Test Accuracy':        [accuracy_score(y_test, y_pred_rf),
                             accuracy_score(y_test, y_pred_xgb)],
    'Test macro F1':        [f1_score(y_test, y_pred_rf,  average='macro'),
                             f1_score(y_test, y_pred_xgb, average='macro')],
})
print(summary.to_string(index=False))
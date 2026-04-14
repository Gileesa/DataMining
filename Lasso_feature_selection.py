import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Parameter
ALPHA_VALUE = 0.1
DATE_COL    = 'date'
TEST_RATIO  = 0.3

# Load the dataset
df = pd.read_csv("DataMining/csv_files/mood_window_dataset.csv")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Target engineering: convert continuous mood to categorical classes for classification
def mood_to_classification(mood):
    if mood <= 3:
        return 'low'
    elif mood <= 6:
        return 'medium'
    else:
        return 'high'

df['target_classification'] = df['target'].apply(mood_to_classification)

# Time-aware split per user
def split_id_per_time(df, test_ratio=TEST_RATIO, date_col=DATE_COL):
    """
    For each user: sort by date, take first rows as train,
    last test_ratio rows as test, in order  data leakage across time.
    """
    train_list = []
    test_list = []

    for user_id, user_df in df.groupby('id'):
        user_df = user_df.sort_values(date_col).reset_index(drop=True)
        split_idx = int((1 - test_ratio) * len(user_df))

        train_list.append(user_df.iloc[:split_idx])
        test_list.append(user_df.iloc[split_idx:])

    train_df = pd.concat(train_list, ignore_index=True).sort_values(date_col).reset_index(drop=True)
    test_df  = pd.concat(test_list,  ignore_index=True).sort_values(date_col).reset_index(drop=True)
    return train_df, test_df

train_df, test_df = split_id_per_time(df)

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
print(f"Train date range: {train_df[DATE_COL].min()} → {train_df[DATE_COL].max()}")
print(f"Test  date range: {test_df[DATE_COL].min()}  → {test_df[DATE_COL].max()}")

# Feature matrix
META_COLS = ['id', 'date', 'target', 'target_classification']
FEATURE_COLS = []

for col in df.columns:
    if col not in META_COLS:
        FEATURE_COLS.append(col)

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']          # continuous target for Lasso
X_test  = test_df[FEATURE_COLS]
y_test  = test_df['target']

# Standardize features for Lasso
scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      

# Lasso feature selection with cross-validation to find optimal alpha
lasso = Lasso(alpha=ALPHA_VALUE, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

feature_importance = pd.Series(lasso.coef_, index=FEATURE_COLS)
print("\nFeature importance scores (sorted):")
print(feature_importance.sort_values(ascending=False).to_string())

selected_features = []
for feature, importance in feature_importance.items():
    if importance != 0:
        selected_features.append(feature)

print(f"\nSelected {len(selected_features)} features:")
print(selected_features)

# Save the selected features
train_df[META_COLS + selected_features].to_csv("DataMining/csv_files/mood_train.csv", index=False)
test_df[META_COLS + selected_features].to_csv("DataMining/csv_files/mood_test.csv", index=False)
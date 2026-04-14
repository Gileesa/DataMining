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
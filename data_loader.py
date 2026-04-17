import pandas as pd
from config import DATE_COL, TRAIN_PATH, TEST_PATH, META_COLS


# Data loading
def load_data(train_path= TRAIN_PATH, test_path= TEST_PATH):
    """
    Load train and test datasets.
    
    return:
    - full train/test dataframes
    - X_train, X_test
    - feature column names
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])

    feature_col = []

    for col in train_df.columns:
        if col not in META_COLS:
            feature_col.append(col)

    X_train = train_df[feature_col].values
    X_test  = test_df[feature_col].values

    return train_df, test_df, X_train, X_test, feature_col
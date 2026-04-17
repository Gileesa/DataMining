import pandas as pd
from config import DATE_COL, TEST_RATIO, TRAIN_PATH, TEST_PATH, LOW_MOOD_MAX, MEDIUM_MOOD_MAX

# Load the dataset
df = pd.read_csv("DataMining/csv_files/KNN/KNN_one_size_mood_window_dataset.csv")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

def mood_to_classification(mood):

    if mood <= LOW_MOOD_MAX:
        return 'low'
    elif mood <= MEDIUM_MOOD_MAX:
        return 'medium'
    else:
        return 'high'

# Apply the classification mapping low = 1, medium = 2, high = 3
df['target_classification'] = df['target'].apply(mood_to_classification)

print("Overall class distribution:")
print(df["target_classification"].value_counts())

# Time-aware split per user
def split_id_per_time(df, test_ratio=TEST_RATIO, date_col = DATE_COL):
    """
    For each user: sort by date, take first rows as train, last test_ratio rows as test.
    """
    train_list = []
    test_list = []

    for user_id, user_df in df.groupby('id'):
        user_df = user_df.sort_values(date_col).reset_index(drop=True)
        split_idx = int((1 - test_ratio) * len(user_df))

        train_list.append(user_df.iloc[:split_idx])
        test_list.append(user_df.iloc[split_idx:])

    # Stack df and  new row numbers
    train_df = pd.concat(train_list, ignore_index=True)

    test_df = pd.concat(test_list, ignore_index=True)
    return train_df, test_df

train_df, test_df = split_id_per_time(df)

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
print(f"Train date range: {train_df[DATE_COL].min()} - {train_df[DATE_COL].max()}")
print(f"Test  date range: {test_df[DATE_COL].min()}  - {test_df[DATE_COL].max()}")


print("\nTrain class distribution:")
print(train_df["target_classification"].value_counts())

print("\nTest class distribution:")
print(test_df["target_classification"].value_counts())

# Save split datasets
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)




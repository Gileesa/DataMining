
import pandas as pd

def test_train_splitting(
    df,
    test_ratio=0.3,
    date_col='date',
    save_prefix='csv_files/mood_dataset'
):
    """
    Splits dataset into train/test using time-based split (per ID).
    
    - First 70% of time → train
    - Last 30% → test
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train_list = []
    test_list = []

    print("\n==== TRAIN / TEST SPLIT INFO ====")

    for uid, user_df in df.groupby('id'):
        user_df = user_df.sort_values(date_col)

        n = len(user_df)
        split_idx = int((1 - test_ratio) * n)

        train_df = user_df.iloc[:split_idx]
        test_df = user_df.iloc[split_idx:]

        train_list.append(train_df)
        test_list.append(test_df)

        print(f"- ID {uid}: train={len(train_df)}, test={len(test_df)}")

    train_final = pd.concat(train_list, ignore_index=True)
    test_final = pd.concat(test_list, ignore_index=True)

    # ---- SAVE ----
    train_path = f"{save_prefix}_train.csv"
    test_path = f"{save_prefix}_test.csv"

    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    print(f"\nSaved train set to: {train_path}")
    print(f"Saved test set to:  {test_path}")
    print(f"Train shape: {train_final.shape}")
    print(f"Test shape:  {test_final.shape}")

    return train_final, test_final


def load_window_dataset(path='csv_files/mood_window_dataset.csv'):
    """
    Load the mood window dataset from CSV.
    """

    df = pd.read_csv(path)

    # ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded dataset from: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


df_windows = load_window_dataset()

train_df, test_df = test_train_splitting(
    df_windows,
    test_ratio=0.3
)
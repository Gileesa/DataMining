import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def categorize_targets(y_array, low_max=6.0, med_max=7.5):
    y_class = np.zeros_like(y_array) # Maak array vol nullen (alles is 'low' default)
    y_class[(y_array > low_max) & (y_array <= med_max)] = 1 # Medium
    y_class[y_array > med_max] = 2 # High
    return y_class

def prepare_rnn_data(df_clean, target_col='mood_avg', window_size=5, test_ratio=0.2):
    """
    Transfrms the cleaned dataframe into 3D tensors suitable for RNN input, using a sliding window approach.
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    
    # getting feature columns (excluding id, date, target, and target_classification)
    feature_cols = [col for col in df_clean.columns if col not in ['id', 'date']]
    target_idx = feature_cols.index(target_col)
    
    # Iterate over each user to create time-aware splits and build the 3D tensors
    for uid, user_data in df_clean.groupby('id'):
        
        
        user_data = user_data.sort_values('date').reset_index(drop=True)
        features = user_data[feature_cols].values
        
        
        split_idx = int(len(features) * (1 - test_ratio))
        
        # Check if we have enough data for the given window size in both train and test sets
        if split_idx <= window_size or (len(features) - split_idx) <= window_size:
            continue
            
        train_features = features[:split_idx]
        test_features = features[split_idx:]
        
        # normalize features (fit on train, transform on both)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        
        # 4. Building the 3D Tensors (Sliding Window) for the Train set
        for i in range(len(train_scaled) - window_size):
            X_train.append(train_scaled[i : i + window_size, :])
            # Target is de mood op de dag exact ná het window
            y_train.append(train_scaled[i + window_size, target_idx])
            
        # 5. Building the 3D Tensors (Sliding Window) for the Test set
        for i in range(len(test_scaled) - window_size):
            X_test.append(test_scaled[i : i + window_size, :])
            y_test.append(test_scaled[i + window_size, target_idx])
            
    # Change to numpy arrays
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), feature_cols
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

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

data = pd.read_csv("(RNN)KNN_one_size_mood_window_dataset.csv")

data_clean = data.drop(columns=['target'])

X_train, y_train, X_test, y_test, feature_names = prepare_rnn_data(
    df_clean=data_clean, 
    target_col='mood_avg', 
    window_size=5, 
    test_ratio=0.3  
)

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Aantal features: {len(feature_names)}")



# Define model parameters based on the shape of the data
aantal_timesteps = X_train.shape[1]  # Dit is 5 (je window size)
aantal_features = X_train.shape[2]   # Dit is 10 (je variabelen minus de target)

# 2. Build the LSTM model
model = Sequential()

# TH LSTM layer
model.add(LSTM(units=32, activation='relu', input_shape=(aantal_timesteps, aantal_features)))

# Dropout 
model.add(Dropout(0.2))

# Output laag:
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model architecture
print("=== Model Architectuur ===")
model.summary()

# Training the model 
print("\nStart training...")
history = model.fit(
    X_train, y_train,
    epochs=30,             
    batch_size=16,         
    validation_data=(X_test, y_test), 
    verbose=1              
)

# 5. Visualize the training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Test Loss (MSE)', color='orange')
plt.title('Leercurve of the LSTM Model')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
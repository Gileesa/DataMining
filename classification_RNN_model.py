from rnn_utils import prepare_rnn_data, categorize_targets
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("(RNN)KNN_one_size_mood_window_dataset.csv")

# 2. Use utility function to prepare data for RNN and categorize targets for classification
X_train, y_train, X_test, y_test, features = prepare_rnn_data(df)
y_train_class = categorize_targets(y_train)
y_test_class = categorize_targets(y_test)


# 3. Model architecture
model_class = Sequential()

# Decide on amount of timesteps and features based on the shape of X_train
aantal_timesteps = X_train.shape[1] 
aantal_features = X_train.shape[2]

# LSTM laYer
model_class.add(LSTM(units=32, activation='relu', input_shape=(aantal_timesteps, aantal_features)))
model_class.add(Dropout(0.2)) # Voorkom overfitting

# Output layer for classification (3 classes: low, medium, high)
model_class.add(Dense(units=3, activation='softmax'))


# Compile the model with appropriate loss and metrics for classification
model_class.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("=== Classificatie Model Architectuur ===")
model_class.summary()


# Training the model
print("\nStart trainen van de Classificatie RNN...")
history_class = model_class.fit(
    X_train, y_train_class,             
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test_class), 
    verbose=1
)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# plot 1: Accuracy 
ax1.plot(history_class.history['accuracy'], label='Train Accuracy', color='blue')
ax1.plot(history_class.history['val_accuracy'], label='Test Accuracy', color='orange')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# plot 2: Loss 
ax2.plot(history_class.history['loss'], label='Train Loss', color='blue')
ax2.plot(history_class.history['val_loss'], label='Test Loss', color='orange')
ax2.set_title('Model Error Rate (Crossentropy)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()
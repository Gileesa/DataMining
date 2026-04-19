# config.py

# Original dataset
DATASET_ORIGIN = 'Datamining/csv_files/dataset_mood_smartphone.csv'                     

# Classification - DataMining/2A/classification_dataset.py
DATE_COL = "date"
ID_COL = "id"
TARGET_COL = "target"
CLASS_COL = "target_classification"

TEST_RATIO = 0.3
WINDOW_SIZE = 5
RANDOM_STATE = 42

LOW_MOOD_MAX = 6
MEDIUM_MOOD_MAX = 7

THREE_D_TENSOR_DATASET_PATH = "csv_files/KNN/(RNN)KNN_one_size_mood_window_dataset.csv"
WINDOW_DATASET_PATH = "Datamining/csv_files/KNN/KNN_one_size_mood_window_dataset.csv"
TRAIN_PATH = "Datamining/csv_files/KNN/mood_train.csv"
TEST_PATH = "Datamining/csv_files/KNN/mood_test.csv"

# Training and Testing split - DataMining/data_loader.py
META_COLS = ['id', 'date', 'target', 'target_classification']

# Random Forest DataMining/random_forest.py
N_SPLITS   = 5 
RANDOM_STATE = 42

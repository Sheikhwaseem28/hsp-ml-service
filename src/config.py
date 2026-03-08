# ml-service/src/config.py
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(PROJECT_DIR)

# Data paths
RAW_DATA_FILE = os.path.join(DATA_DIR, 'data.csv')
OUTPUT_DATA_FILE = os.path.join(DATA_DIR, 'output.csv')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data')

PROCESSED_DATA_FILES = {
    'X_train': os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'),
    'X_test': os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'),
    'y_train': os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'),
    'y_test': os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'),
}

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILES = {
    'model': os.path.join(MODEL_DIR, 'model.joblib'),
    'metadata': os.path.join(MODEL_DIR, 'metadata.json'),
    'feature_columns': os.path.join(MODEL_DIR, 'feature_columns.joblib'),
    'scaler': os.path.join(MODEL_DIR, 'scaler.joblib'),
}

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

# Ensure directories exist
for directory in [PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

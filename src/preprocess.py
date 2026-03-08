# ml-service/src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .config import RAW_DATA_FILE, PROCESSED_DATA_FILES, MODEL_FILES
from .utils import logger

def preprocess_data():
    """Load, clean, and preprocess data from data.csv"""
    logger.info(f"Loading raw data from {RAW_DATA_FILE}")
    
    if not os.path.exists(RAW_DATA_FILE):
        logger.error(f"Raw data file not found at {RAW_DATA_FILE}")
        return
    
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Feature engineering/selection
    # In a real scenario, we'd do more exploratory data analysis
    
    # Basic data cleaning
    df = df.dropna()
    
    # Feature selection (dropping non-numeric or less relevant columns for a basic model)
    target = 'price'
    
    # Log transform the price to handle skewness
    df[target] = np.log1p(df[target])
    
    # Define features
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated'
    ]
    
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, MODEL_FILES['scaler'])
    logger.info(f"Scaler saved to {MODEL_FILES['scaler']}")
    
    # Save feature columns
    joblib.dump(features, MODEL_FILES['feature_columns'])
    logger.info(f"Feature columns saved to {MODEL_FILES['feature_columns']}")
    
    # Save processed data
    np.save(PROCESSED_DATA_FILES['X_train'], X_train_scaled)
    np.save(PROCESSED_DATA_FILES['X_test'], X_test_scaled)
    np.save(PROCESSED_DATA_FILES['y_train'], y_train.values)
    np.save(PROCESSED_DATA_FILES['y_test'], y_test.values)
    
    logger.info("Data preprocessing completed successfully")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data()

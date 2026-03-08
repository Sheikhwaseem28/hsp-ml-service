# ml-service/src/predict.py
import joblib
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to sys.path to allow absolute imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_FILES
from src.utils import logger

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.load_model()

    def load_model(self):
        """Load the trained model, scaler, and feature list"""
        try:
            if os.path.exists(MODEL_FILES['model']):
                self.model = joblib.load(MODEL_FILES['model'])
                self.scaler = joblib.load(MODEL_FILES['scaler'])
                self.features = joblib.load(MODEL_FILES['feature_columns'])
                logger.info("Model and associated files loaded successfully")
            else:
                logger.warning(f"Model file not found at {MODEL_FILES['model']}. Please train the model first.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def predict(self, input_data):
        """Predict the price for a single house or a batch of houses"""
        if self.model is None:
            return {"error": "Model not loaded. Please train the model."}

        try:
            # Convert input to DataFrame if it's a list/dict
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)

            # Reorder columns to match training features and handle missing columns
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0
            
            df = df[self.features]

            # Scale the input
            X_scaled = self.scaler.transform(df)

            # Predict
            log_predictions = self.model.predict(X_scaled)
            
            # Convert back from log scale
            predictions = np.expm1(log_predictions)

            return predictions.tolist()
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}

# Singleton instance
predictor = HousePricePredictor()

def get_prediction(data):
    return predictor.predict(data)

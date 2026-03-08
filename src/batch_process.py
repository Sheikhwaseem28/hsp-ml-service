# ml-service/src/batch_process.py
import pandas as pd
import os
from .predict import predictor
from .config import RAW_DATA_FILE, OUTPUT_DATA_FILE
from .utils import logger

def process_csv():
    """Read a CSV file, predict prices, and save to output.csv"""
    if not os.path.exists(RAW_DATA_FILE):
        logger.error(f"Input file not found: {RAW_DATA_FILE}")
        return

    logger.info(f"Reading data from {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # We only predict for a subset if the file is large, but here we'll do all
    logger.info(f"Processing {len(df)} rows...")
    
    # Prepare data for prediction
    # predictor.predict handles the feature selection/scaling
    predictions = predictor.predict(df.to_dict(orient='records'))
    
    if isinstance(predictions, dict) and 'error' in predictions:
        logger.error(f"Batch processing error: {predictions['error']}")
        return

    # Add predictions to the dataframe
    df['predicted_price'] = predictions
    
    # Save to output.csv
    df.to_csv(OUTPUT_DATA_FILE, index=False)
    logger.info(f"Predictions saved to {OUTPUT_DATA_FILE}")

if __name__ == "__main__":
    process_csv()

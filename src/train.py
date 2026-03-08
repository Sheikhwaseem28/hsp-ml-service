# ml-service/src/train.py
import numpy as np
import pandas as pd
import json
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from .config import PROCESSED_DATA_FILES, MODEL_FILES, MODEL_PARAMS
from .utils import logger, save_model_metadata, NumpyEncoder

def load_preprocessed_data():
    """Load preprocessed data"""
    X_train = np.load(PROCESSED_DATA_FILES['X_train'], allow_pickle=True)
    X_test = np.load(PROCESSED_DATA_FILES['X_test'], allow_pickle=True)
    y_train = np.load(PROCESSED_DATA_FILES['y_train'], allow_pickle=True)
    y_test = np.load(PROCESSED_DATA_FILES['y_test'], allow_pickle=True)
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        f'{prefix}rmse': float(rmse),
        f'{prefix}mae': float(mae),
        f'{prefix}r2': float(r2),
        f'{prefix}mse': float(mse)
    }
    
    return metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    logger.info("Training XGBoost model...")
    
    model = XGBRegressor(
        n_estimators=MODEL_PARAMS['n_estimators'],
        max_depth=MODEL_PARAMS['max_depth'],
        learning_rate=MODEL_PARAMS['learning_rate'],
        random_state=MODEL_PARAMS['random_state'],
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert from log scale
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual, y_pred_train_actual, 'train_')
    test_metrics = calculate_metrics(y_test_actual, y_pred_test_actual, 'test_')
    
    metrics = {**train_metrics, **test_metrics}
    
    logger.info(f"XGBoost - Test RMSE: ${metrics['test_rmse']:,.2f}")
    logger.info(f"XGBoost - Test R²: {metrics['test_r2']:.4f}")
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    logger.info("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=MODEL_PARAMS['n_estimators'],
        max_depth=MODEL_PARAMS['max_depth'],
        random_state=MODEL_PARAMS['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert from log scale
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual, y_pred_train_actual, 'train_')
    test_metrics = calculate_metrics(y_test_actual, y_pred_test_actual, 'test_')
    
    metrics = {**train_metrics, **test_metrics}
    
    logger.info(f"Random Forest - Test RMSE: ${metrics['test_rmse']:,.2f}")
    logger.info(f"Random Forest - Test R²: {metrics['test_r2']:.4f}")
    
    return model, metrics

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model"""
    logger.info("Training Gradient Boosting model...")
    
    model = GradientBoostingRegressor(
        n_estimators=MODEL_PARAMS['n_estimators'],
        max_depth=MODEL_PARAMS['max_depth'],
        learning_rate=MODEL_PARAMS['learning_rate'],
        random_state=MODEL_PARAMS['random_state']
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert from log scale
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual, y_pred_train_actual, 'train_')
    test_metrics = calculate_metrics(y_test_actual, y_pred_test_actual, 'test_')
    
    metrics = {**train_metrics, **test_metrics}
    
    logger.info(f"Gradient Boosting - Test RMSE: ${metrics['test_rmse']:,.2f}")
    logger.info(f"Gradient Boosting - Test R²: {metrics['test_r2']:.4f}")
    
    return model, metrics

def train_ridge(X_train, y_train, X_test, y_test):
    """Train Ridge regression model"""
    logger.info("Training Ridge regression model...")
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert from log scale
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual, y_pred_train_actual, 'train_')
    test_metrics = calculate_metrics(y_test_actual, y_pred_test_actual, 'test_')
    
    metrics = {**train_metrics, **test_metrics}
    
    logger.info(f"Ridge - Test RMSE: ${metrics['test_rmse']:,.2f}")
    logger.info(f"Ridge - Test R²: {metrics['test_r2']:.4f}")
    
    return model, metrics

def select_best_model(models_metrics):
    """Select the best model based on test RMSE"""
    best_model_name = None
    best_rmse = float('inf')
    best_model = None
    best_metrics = None
    
    for name, (model, metrics) in models_metrics.items():
        if metrics['test_rmse'] < best_rmse:
            best_rmse = metrics['test_rmse']
            best_model_name = name
            best_model = model
            best_metrics = metrics
    
    return best_model_name, best_model, best_metrics

def train_and_save_model():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Load feature columns
    feature_columns = joblib.load(MODEL_FILES['feature_columns'])
    
    # Train multiple models
    models_metrics = {}
    
    # XGBoost
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models_metrics['XGBoost'] = (model, metrics)
    
    # Random Forest
    model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models_metrics['RandomForest'] = (model, metrics)
    
    # Gradient Boosting
    model, metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models_metrics['GradientBoosting'] = (model, metrics)
    
    # Ridge
    model, metrics = train_ridge(X_train, y_train, X_test, y_test)
    models_metrics['Ridge'] = (model, metrics)
    
    # Select best model
    best_model_name, best_model, best_metrics = select_best_model(models_metrics)
    
    logger.info("=" * 60)
    logger.info(f"🏆 Best Model: {best_model_name}")
    logger.info(f"   Test RMSE: ${best_metrics['test_rmse']:,.2f}")
    logger.info(f"   Test MAE: ${best_metrics['test_mae']:,.2f}")
    logger.info(f"   Test R²: {best_metrics['test_r2']:.4f}")
    logger.info("=" * 60)
    
    # Save best model
    joblib.dump(best_model, MODEL_FILES['model'])
    
    # Save metadata
    metadata = save_model_metadata(
        best_model_name,
        best_metrics,
        feature_columns,
        MODEL_PARAMS
    )
    
    with open(MODEL_FILES['metadata'], 'w') as f:
        json.dump(metadata, f, cls=NumpyEncoder, indent=2)
    
    logger.info(f"Model saved to {MODEL_FILES['model']}")
    logger.info(f"Metadata saved to {MODEL_FILES['metadata']}")
    
    return best_model, best_metrics, feature_columns

if __name__ == "__main__":
    train_and_save_model()
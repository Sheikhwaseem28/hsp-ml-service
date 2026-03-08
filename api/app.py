# ml-service/api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_prediction
from src.utils import logger
import json

app = Flask(__name__)

# Allow all origins (backend on Vercel, local dev, etc.)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "ml-service"})

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "No features provided"}), 400
    
    prediction = get_prediction(data['features'])
    
    if isinstance(prediction, dict) and 'error' in prediction:
        return jsonify(prediction), 500
        
    return jsonify({"prediction": prediction[0]})

@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'properties' not in data:
        return jsonify({"error": "No properties provided"}), 400
    
    predictions = get_prediction(data['properties'])
    
    if isinstance(predictions, dict) and 'error' in predictions:
        return jsonify(predictions), 500
        
    return jsonify({"predictions": predictions})

@app.route('/api/v1/features', methods=['GET'])
def get_features():
    from src.config import MODEL_FILES
    import joblib
    try:
        features = joblib.load(MODEL_FILES['feature_columns'])
        return jsonify({"required_features": features})
    except:
        return jsonify({"error": "Features list not found"}), 404

@app.route('/api/v1/model/info', methods=['GET'])
def get_model_info():
    from src.config import MODEL_FILES
    try:
        with open(MODEL_FILES['metadata'], 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except:
        return jsonify({"error": "Model metadata not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting ML Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

# ml-service/src/utils.py
import logging
import json
import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml-service')

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_model_metadata(model_name, metrics, feature_columns, params):
    """Prepare model metadata for saving"""
    metadata = {
        'model_name': model_name,
        'trained_at': datetime.datetime.now().isoformat(),
        'metrics': metrics,
        'features': feature_columns,
        'hyperparameters': params
    }
    return metadata

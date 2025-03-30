# Configure matplotlib first to prevent threading issues
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import sys

# Make sure the models directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the model functions
from models.linReg import run_linear_regression
from models.knn import predict_single_point

# Import the sample data generator
try:
    from datasets.sample_data import generate_linear_data
except ImportError:
    # Create the datasets directory if it doesn't exist
    os.makedirs(os.path.join(current_dir, 'datasets'), exist_ok=True)
    
    # Create a simple sample_data.py file if it doesn't exist
    sample_data_path = os.path.join(current_dir, 'datasets', 'sample_data.py')
    if not os.path.exists(sample_data_path):
        with open(sample_data_path, 'w') as f:
            f.write('''import numpy as np

def generate_linear_data(n_samples=100, noise=10.0, seed=42):
    """
    Generate sample data for linear regression with controlled noise
    
    Parameters:
    n_samples (int): Number of samples to generate
    noise (float): Amount of noise to add
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary with 'X' and 'y' keys containing the data
    """
    np.random.seed(seed)
    
    # Generate X values
    X = np.linspace(0, 10, n_samples)
    
    # True relationship: y = 2x + 5 + noise
    true_coef = 2.0
    true_intercept = 5.0
    
    # Generate y values with noise
    y = true_coef * X + true_intercept + np.random.normal(0, noise, n_samples)
    
    return {
        'X': X.tolist(),
        'y': y.tolist(),
        'true_coef': true_coef,
        'true_intercept': true_intercept
    }
''')
    
    # Create an __init__.py file in the datasets directory
    with open(os.path.join(current_dir, 'datasets', '__init__.py'), 'w') as f:
        f.write('# Datasets package initialization\n')
    
    # Now try to import again
    from datasets.sample_data import generate_linear_data

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Health check endpoint - make sure this matches what the frontend is calling
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# Debug route to test API connectivity
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "ML Visualizer API is running",
        "endpoints": [
            "/api/health",
            "/api/linear-regression",
            "/api/linear-regression/sample",
            "/api/knn-classification",
            "/api/knn-regression"
        ]
    })

@app.route('/api/linear-regression', methods=['POST'])
def linear_regression():
    data = request.json
    
    try:
        # Log incoming data for debugging
        print(f"Received API request for linear regression")
        print(f"Request data type: {type(data)}")
        print(f"Request data content: {data}")
        
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
            
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
            
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
            
        if len(data['X']) < 3:
            return jsonify({"error": "At least 3 data points are required for regression"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Get alpha parameter with default value
        alpha = data.get('alpha', 0.01)
        
        # Get iterations parameter with default value
        iterations = data.get('iterations', 100)
        
        # Ensure iterations is an integer
        try:
            iterations = int(iterations)
            if iterations < 1:
                iterations = 100  # Default if invalid
        except (ValueError, TypeError):
            iterations = 100  # Default if invalid

         # Call the model with alpha and iterations parameters
        print(f"Calling linear regression model function with alpha={alpha}, iterations={iterations}")
        result = run_linear_regression(data, alpha=alpha, iterations=iterations)
        
        # Check if the result is None (which would cause JSON serialization issues)
        if result is None:
            return jsonify({"error": "Model returned None"}), 500
            
        # Check for error field    
        if isinstance(result, dict) and 'error' in result:
            print(f"Error in linear regression: {result['error']}")
            return jsonify(result), 400
        
        # Ensure the response is JSON serializable
        if not isinstance(result, dict):
            return jsonify({"error": f"Expected dict result, got {type(result)}"}), 500
            
        # Create a clean copy with only essential fields
        sanitized_result = {
            'coefficient': float(result.get('coefficient', 0)),
            'intercept': float(result.get('intercept', 0)),
            'mse': float(result.get('mse', 0)),  # Change from mse_train
            'r2': float(result.get('r2', 0)),    # Change from r2_train
            'equation': result.get('equation', 'y = 0x + 0')
        }
        
        # Add plot only if available and valid
        if 'plot' in result and result['plot']:
            sanitized_result['plot'] = result['plot']
            
        print(f"Returning successful result with keys: {list(sanitized_result.keys())}")
        return jsonify(sanitized_result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in linear regression endpoint: {str(e)}")
        print(f"Traceback: {error_details}")
        
        error_response = {
            "error": str(e),
            "traceback": error_details
        }
        
        # Add request data info if available
        if data and isinstance(data, dict):
            error_response["data_info"] = {
                "X_type": str(type(data.get('X', None))),
                "y_type": str(type(data.get('y', None))),
                "X_length": len(data.get('X', [])) if isinstance(data.get('X', None), list) else "Not a list",
                "y_length": len(data.get('y', [])) if isinstance(data.get('y', None), list) else "Not a list"
            }
            
        return jsonify(error_response), 500

@app.route('/api/linear-regression/sample', methods=['GET'])
def linear_regression_sample():
    try:
        n_samples = request.args.get('n_samples', default=30, type=int)
        noise = request.args.get('noise', default=5.0, type=float)
        data = generate_linear_data(n_samples=n_samples, noise=noise)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# KNN endpoints
@app.route('/api/knn-classification', methods=['POST'])
def knn_classification():
    data = request.json
    try:
        n_neighbors = data.get('n_neighbors', 5)
        result = run_knn_classification(data, n_neighbors=n_neighbors)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/knn-regression', methods=['POST'])
def knn_regression():
    data = request.json
    try:
        n_neighbors = data.get('n_neighbors', 5)
        result = run_knn_regression(data, n_neighbors=n_neighbors)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Add these new endpoints after your existing KNN endpoints

# Keep all existing endpoints and modify the knn-predict-point endpoint

@app.route('/api/knn-predict-point', methods=['POST'])
def knn_predict_point():
    data = request.json
    try:
        # Extract the point to predict
        predict_point = data.pop('predict_point', None)
        if not predict_point:
            return jsonify({"error": "No prediction point provided"}), 400
            
        n_neighbors = data.get('n_neighbors', 5)
        
        # Get predictions
        from models.knn import predict_single_point
        result = predict_single_point(data, predict_point, n_neighbors)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Remove the knn-interactive-train endpoint as we're not generating boundaries



if __name__ == '__main__':
    print(" * ML Visualizer Backend Starting...")
    print(" * Make sure you have the required packages installed")
    print(" * Backend will be available at http://localhost:5000")
    print(" * Test the API by accessing http://localhost:5000/ in your browser")
    print(" * Press Ctrl+C to stop the server")
    # Try using a different host format if 0.0.0.0 isn't working
    app.run(host='127.0.0.1', port=5000, debug=True)

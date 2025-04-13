import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import numpy as np
from flask import Flask, jsonify, request # type: ignore
from flask_cors import CORS # type: ignore
import os
import sys

app = Flask(__name__)

# Update CORS configuration to explicitly allow your Firebase domain
CORS(app, resources={
    r"/*": {
        "origins": ["https://algovizz.web.app", "http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the model functions
from models.Reg import run_polynomial_regression
from models.knn import predict_single_point, generate_decision_boundary
from models.kmeans import run_kmeans, generate_clustering_data
from models.PCA import run_pca, generate_pca_data  
from models.DTrees import (run_decision_tree_classification, run_decision_tree_regression, generate_tree_visualization,
                          predict_data_points, generate_sample_classification_data, generate_optimized_decision_boundary,
                          generate_sample_regression_data, generate_tree_with_highlighted_path, generate_regression_surface)
from models.SVM import run_svm
from models.ANN import run_ann, predict_points, generate_sample_data

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from datasets.sample_data import generate_linear_data

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
            "/api/knn-regression",
            "/api/kmeans",              
            "/api/kmeans/sample",
            "/api/pca",                  
            "/api/pca/sample-data",
            "/api/dbscan"  
        ]
    })

@app.route('/api/svm', methods=['POST'])
def svm():
    data = request.json
    try:
        # Log incoming data for debugging
        print(f"Received API request for SVM")
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
            return jsonify({"error": "At least 3 data points are required for SVM"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Check if we're predicting new points
        if 'predict' in data and data['predict']:
            if 'model' not in data:
                return jsonify({"error": "Model must be provided for prediction"}), 400
                
            # Call the prediction function
            result = predict_new_points(data['model'], data['predict'])
        else:
            # Call the SVM model training function
            result = run_svm(data)

        if result is None:
            return jsonify({"error": "Model returned None"}), 500
        
        # Check for error field    
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in SVM endpoint: {str(e)}")
        print(f"Traceback: {error_details}")
        
        return jsonify({"error": str(e), "traceback": error_details}), 500

@app.route('/api/svm/sample', methods=['GET'])
def svm_sample():
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
        result = predict_single_point(data, n_neighbors=n_neighbors)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/knn-regression', methods=['POST'])
def knn_regression():
    data = request.json
    try:
        n_neighbors = data.get('n_neighbors', 5)
        result = generate_decision_boundary(data, n_neighbors=n_neighbors)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/knn-predict-point', methods=['POST'])
def knn_predict_point():
    data = request.json
    try:
        # Extract the point to predict
        predict_point = data.pop('predict_point', None)
        if not predict_point:
            return jsonify({"error": "No prediction point provided"}), 400
            
        n_neighbors = data.get('n_neighbors', 5)
        
        # Print debug info
        print(f"KNN predict point request received")
        print(f"X sample: {data['X'][:3]}")
        print(f"y sample: {data['y'][:10]}")
        print(f"y types: {[type(y) for y in data['y'][:5]]}")
        print(f"Predict point: {predict_point}")
        print(f"n_neighbors: {n_neighbors}")
        
        # Validate inputs
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
        
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
        
        if len(data['X']) < 1:
            return jsonify({"error": "At least 1 data point is required for prediction"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Get predictions
        result = predict_single_point(data, predict_point, n_neighbors)
        
        # Print result
        print(f"Prediction result: {result}")
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Exception in KNN predict point endpoint: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500



@app.route('/api/knn-decision-boundary', methods=['POST'])
def knn_decision_boundary():
    data = request.json
    try:
        n_neighbors = data.get('n_neighbors', 5)
        
        # Validate inputs
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
        
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
        
        if len(data['X']) < 5:
            return jsonify({"error": "At least 5 data points are required for decision boundary"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Generate decision boundary
        result = generate_decision_boundary(data, n_neighbors)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Exception in KNN decision boundary endpoint: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

@app.route('/api/kmeans', methods=['POST'])
def kmeans_clustering():
    data = request.json
    
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
            
        if 'X' not in data:
            return jsonify({"error": "Missing required data field: X must be provided"}), 400
            
        if not isinstance(data['X'], list):
            return jsonify({"error": "X must be an array/list"}), 400
            
        if len(data['X']) < 3:
            return jsonify({"error": "At least 3 data points are required for clustering"}), 400
        
        # Get parameters with default values
        k = data.get('k', 3)
        max_iterations = data.get('max_iterations', 100)
        
        # Run K-means
        result = run_kmeans(data, k=k, max_iterations=max_iterations)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in K-means endpoint: {str(e)}")
        print(f"Traceback: {error_details}")
        
        return jsonify({
            "error": str(e),
            "traceback": error_details
        }), 500

@app.route('/api/kmeans/sample', methods=['GET'])
def kmeans_sample_data():
    try:
        n_samples = request.args.get('n_samples', default=100, type=int)
        n_clusters = request.args.get('n_clusters', default=3, type=int)
        variance = request.args.get('variance', default=0.5, type=float)
        dataset_type = request.args.get('dataset_type', default='blobs', type=str)
        
        data = generate_clustering_data(
            n_samples=n_samples,
            n_clusters=n_clusters,
            variance=variance,
            dataset_type=dataset_type
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/kmeans/preview', methods=['POST'])
def kmeans_preview():
    data = request.json
    
    try:
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
            
        if 'X' not in data:
            return jsonify({"error": "Missing required data field: X must be provided"}), 400
            
        if not isinstance(data['X'], list):
            return jsonify({"error": "X must be an array/list"}), 400
            
        if len(data['X']) < 1:
            return jsonify({"error": "At least 1 data point is required for preview"}), 400
        
        # Generate a simple scatter plot of the data
        import matplotlib.pyplot as plt
        import numpy as np
        import base64
        from io import BytesIO
        
        X = np.array(data['X'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], color='#3f51b5', alpha=0.7)
        
        ax.set_title('Data Preview')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(alpha=0.3)
        
        # Add padding around the data
        x_range = X[:, 0].max() - X[:, 0].min()
        y_range = X[:, 1].max() - X[:, 1].min()
        
        padding = 0.1
        ax.set_xlim(X[:, 0].min() - padding * x_range, X[:, 0].max() + padding * x_range)
        ax.set_ylim(X[:, 1].min() - padding * y_range, X[:, 1].max() + padding * y_range)
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)
        buffer.seek(0)
        
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'plot': plot_base64
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in K-means preview endpoint: {str(e)}")
        print(f"Traceback: {error_details}")
        
        return jsonify({
            "error": str(e),
            "traceback": error_details
        }), 500

@app.route('/api/pca', methods=['POST'])
def pca_analysis():
    data = request.json
    
    try:
        # Log incoming data for debugging
        print(f"Received API request for PCA")
        print(f"Request data type: {type(data)}")
        print(f"Request data content: {data}")
        
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
            
        if 'X' not in data:
            return jsonify({"error": "Missing required data field: X must be provided"}), 400
            
        if not isinstance(data['X'], list):
            return jsonify({"error": "X must be an array/list"}), 400
            
        if len(data['X']) < 2:
            return jsonify({"error": "At least 2 data points are required for PCA"}), 400
        
        result = run_pca(data)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"Error in PCA endpoint: {str(e)}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500

@app.route('/api/pca/sample-data', methods=['GET'])
def pca_sample_data():
    try:
        # Get query parameters
        n_samples = request.args.get('n_samples', default=50, type=int)
        noise = request.args.get('noise', default=0.1, type=float)

        # Generate sample data with parameters
        data = generate_pca_data(n_samples=n_samples, noise=noise)

        return jsonify(data)
    except Exception as e:
        import traceback
        print(f"Error generating PCA sample data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/dtree-classification', methods=['POST'])
def dtree_classification():
    data = request.json
    try:
        # Get parameters with default values
        max_depth = data.get('max_depth', 3)
        min_samples_split = data.get('min_samples_split', 2)
        criterion = data.get('criterion', 'gini')
        
        # Validate inputs
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
        
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
        
        if len(data['X']) < 2:
            return jsonify({"error": "At least 2 data points are required for classification"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Run decision tree classification
        result = run_decision_tree_classification(data, max_depth, min_samples_split, criterion)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/dtree-regression', methods=['POST'])
def dtree_regression():
    data = request.json
    try:
        # Get parameters with default values
        max_depth = data.get('max_depth', 3)
        min_samples_split = data.get('min_samples_split', 2)
        
        # Validate inputs
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
        
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
        
        if len(data['X']) < 2:
            return jsonify({"error": "At least 2 data points are required for regression"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Run decision tree regression
        result = run_decision_tree_regression(data, max_depth, min_samples_split)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/dtree/predict', methods=['POST', 'OPTIONS'])
def dtree_predict_new():
    if request.method == 'OPTIONS':
        # CORS preflight response
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    data = request.json
    try:
        # Extract training data and points to predict
        training_data = data.get('trained_points', [])
        predict_points = data.get('predict_points', [])
        
        # Format the data for the model
        X = []
        y = []
        
        if data.get('type') == 'classification':
            for point in training_data:
                X.append([float(point['x1']), float(point['x2'])])
                if 'class' in point:
                    y.append(str(point['class']))
                elif 'y' in point:
                    y.append(str(point['y']))
                    
            # Process prediction points
            pred_X = []
            for point in predict_points:
                pred_X.append([float(point['x1']), float(point['x2'])])
                
            # Get parameters
            max_depth = data.get('parameters', {}).get('max_depth', 3)
            min_samples_split = data.get('parameters', {}).get('min_samples_split', 2)
            criterion = data.get('parameters', {}).get('criterion', 'gini')
            
            # Call the model function
            result = predict_data_points({'X': X, 'y': y}, pred_X, max_depth, min_samples_split, criterion)
            
            # Limit image size to avoid HTTP header size issues
            if 'tree_visualization' in result and len(result['tree_visualization']) > 500000:
                result['tree_visualization'] = result['tree_visualization'][:500000]
                print("Warning: Tree visualization was truncated due to size")
                
            # Convert NumPy types to Python types
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            result = convert_to_python_types(result)
            
            return jsonify(result)
        else:
            # Handle regression similarly
            for point in training_data:
                X.append([float(point['x1']), float(point['x2'])])
                if 'value' in point:
                    y.append(float(point['value']))
                elif 'y' in point:
                    y.append(float(point['y']))
            # Process prediction points
            pred_X = []
            for point in predict_points:
                pred_X.append([float(point['x1']), float(point['x2'])])
                
            # Get parameters
            max_depth = data.get('parameters', {}).get('max_depth', 3)
            min_samples_split = data.get('parameters', {}).get('min_samples_split', 2)
            criterion = data.get('parameters', {}).get('criterion', 'gini')
            
            # Call the model function
            result = predict_data_points({'X': X, 'y': y}, pred_X, max_depth, min_samples_split, criterion)
            
            # Limit image size to avoid HTTP header size issues
            if 'tree_visualization' in result and len(result['tree_visualization']) > 500000:
                result['tree_visualization'] = result['tree_visualization'][:500000]
                print("Warning: Tree visualization was truncated due to size")
                
            # Convert NumPy types to Python types
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            result = convert_to_python_types(result)
            
            return jsonify(result)
                    
            # Rest of regression code similar to above
            # ...
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in dtree prediction: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

@app.route('/api/dtree/sample_data', methods=['POST'])
def dtree_sample_data():
    try:
        data = request.json
        data_type = data.get('type', 'classification')
        count = data.get('count', 40)
        dataset_type = data.get('dataset_type', 'blobs')
        n_clusters = data.get('n_clusters', 3)
        variance = data.get('variance', 0.5)
        sparsity = data.get('sparsity', 1.0)  # New parameter for regression data sparsity
        
        if data_type == 'classification':
            result = generate_sample_classification_data(
                dataset_type=dataset_type,
                n_samples=count,
                n_clusters=n_clusters,
                variance=variance
            )
        else:
            # Use the updated regression function with parameters including sparsity
            result = generate_sample_regression_data(
                dataset_type=dataset_type,
                n_samples=count,
                variance=variance,
                sparsity=sparsity  # Pass the sparsity parameter
            )
                
        # Convert to points format
        points = []
        for i in range(len(result['X'])):
            points.append({
                'x1': result['X'][i][0],
                'x2': result['X'][i][1],
                'y': result['y'][i] if 'y' in result else str(result['values'][i])
            })
            
        return jsonify({'points': points})
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/dtree/sample-regression', methods=['GET'])
def dtree_sample_regression_data():
    try:
        data = generate_sample_regression_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dtree/highlight-path', methods=['POST'])
def dtree_highlight_path():
    data = request.json
    try:
        # Extract required data
        training_data = {'X': data.get('X', []), 'y': data.get('y', [])}
        point = data.get('point', [])
        path_indices = data.get('path_indices', [])
        
        # Get parameters
        max_depth = data.get('max_depth', 3)
        min_samples_split = data.get('min_samples_split', 2)
        criterion = data.get('criterion', 'gini')
        is_regression = data.get('is_regression', False)
        
        # Validate inputs
        if not training_data['X'] or not training_data['y']:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
        
        if not point:
            return jsonify({"error": "No point provided for path highlighting"}), 400
        
        if not path_indices:
            return jsonify({"error": "No path indices provided"}), 400
        
        # Convert data to numpy arrays
        X = np.array(training_data['X'])
        y = np.array(training_data['y'])
        point = np.array(point)
        
        # Train the appropriate model
        if is_regression:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        else:
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42
            )
        
        model.fit(X, y)
        
        # Generate tree visualization with highlighted path
        if is_regression:
            tree_img = generate_tree_with_highlighted_path(
                model, point, path_indices, 
                feature_names=['x1', 'x2'], 
                regression=True
            )
        else:
            tree_img = generate_tree_with_highlighted_path(
                model, point, path_indices, 
                feature_names=['x1', 'x2'], 
                class_names=np.unique(y)
            )
        
        return jsonify({"tree_visualization": tree_img})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/dtree/train', methods=['POST', 'OPTIONS'])
def dtree_train():
    if request.method == 'OPTIONS':
        # CORS preflight response
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    data = request.json
    
    try:
        # Extract training data
        training_data = data.get('trained_points', [])
        
        if len(training_data) < 2:
            return jsonify({"error": "Need at least 2 training points"}), 400
            
        # Get parameters with default values
        max_depth = int(data.get('parameters', {}).get('max_depth', 3))
        min_samples_split = int(data.get('parameters', {}).get('min_samples_split', 2))
        criterion = data.get('parameters', {}).get('criterion', 'gini')
        
        # Separate X and y based on training data structure
        X = []
        y = []
        
        # Print some debug info
        print(f"Training data type: {data.get('type')}")
        print(f"First point: {training_data[0] if training_data else 'No data'}")
        
        if data.get('type') == 'classification':
            for point in training_data:
                try:
                    x1 = float(point['x1'])
                    x2 = float(point['x2'])
                    X.append([x1, x2])
                    
                    # For classification, y can be string or number but we'll convert to string for consistency
                    if 'class' in point:
                        y.append(str(point['class']))
                    elif 'y' in point:
                        y.append(str(point['y']))
                    else:
                        return jsonify({"error": f"Point missing class or y value: {point}"}), 400
                except (ValueError, TypeError) as e:
                    return jsonify({"error": f"Invalid data value in point {point}: {e}"}), 400
                except KeyError as e:
                    return jsonify({"error": f"Missing key in point {point}: {e}"}), 400
            
            
            # Critical: ensure unique classes by manually converting to numeric labels
            unique_classes = list(set(y))
            class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
            y_numeric = [class_to_num[cls] for cls in y]
            
            print(f"Numeric y: {y_numeric}")
            
            # Convert X to numpy array with explicit data type
            X_arr = np.array(X, dtype=np.float64)
            y_arr = np.array(y_numeric, dtype=np.int32)
            
            # Check for NaN or infinity
            if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)):
                return jsonify({"error": "Data contains NaN or infinite values"}), 400
                
            print(f"X array shape: {X_arr.shape}, y array shape: {y_arr.shape}")
            print(f"X array dtype: {X_arr.dtype}, y array dtype: {y_arr.dtype}")
            
            # Run decision tree classification with our numeric y
            result = run_decision_tree_classification({'X': X_arr, 'y': y_arr}, max_depth, min_samples_split, criterion)
            
            # Add class mapping to result for frontend interpretation
            result['class_mapping'] = {str(v): str(k) for k, v in class_to_num.items()}
            
        else:  # Regression case
            for point in training_data:
                try:
                    x1 = float(point['x1'])
                    x2 = float(point['x2'])
                    X.append([x1, x2])
                    
                    # For regression, y must be numeric
                    if 'value' in point:
                        y.append(float(point['value']))
                    elif 'y' in point:
                        y.append(float(point['y']))
                    else:
                        return jsonify({"error": f"Point missing value or y: {point}"}), 400
                except (ValueError, TypeError) as e:
                    return jsonify({"error": f"Invalid numeric value in point {point}: {e}"}), 400
                except KeyError as e:
                    return jsonify({"error": f"Missing key in point {point}: {e}"}), 400
            
            # Convert X and y to numpy arrays with explicit data type
            X_arr = np.array(X, dtype=np.float64)
            y_arr = np.array(y, dtype=np.float64)
            
            # Check for NaN or infinity
            if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)) or np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
                return jsonify({"error": "Data contains NaN or infinite values"}), 400
                
            print(f"X array shape: {X_arr.shape}, y array shape: {y_arr.shape}")
            print(f"X array dtype: {X_arr.dtype}, y array dtype: {y_arr.dtype}")
            
            # Run decision tree regression
            result = run_decision_tree_regression({'X': X_arr, 'y': y_arr}, max_depth, min_samples_split)

        if 'tree_visualization' in result and result['tree_visualization']:
            # Save the base64 image to a temporary file or database
            # For now, just truncate it if it's too large
            if len(result['tree_visualization']) > 500000:  # If over ~500kb
                # Either compress further or just notify it's too large
                result['tree_visualization'] = result['tree_visualization'][:500000]
                print("Warning: Tree visualization was truncated due to size")

        if 'error' in result:
            return jsonify(result), 400
        
        # Convert any NumPy types to Python native types to make it JSON serializable
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Convert all values in the result to Python native types
        result = convert_to_python_types(result)
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in dtree_train: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

@app.route('/api/dtree/visualize', methods=['POST', 'OPTIONS'])
def dtree_visualize():
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    data = request.json
    try:
        # Extract training data
        training_data = data.get('trained_points', [])
        
        if len(training_data) < 2:
            return jsonify({"error": "Need at least 2 training points"}), 400
            
        # Get parameters with default values
        max_depth = int(data.get('parameters', {}).get('max_depth', 3))
        min_samples_split = int(data.get('parameters', {}).get('min_samples_split', 2))
        criterion = data.get('parameters', {}).get('criterion', 'gini')
        
        # Format the data for the model
        X = []
        y = []
        
        if data.get('type') == 'classification':
            for point in training_data:
                try:
                    x1 = float(point['x1'])
                    x2 = float(point['x2'])
                    X.append([x1, x2])
                    
                    if 'class' in point:
                        y.append(str(point['class']))
                    elif 'y' in point:
                        y.append(str(point['y']))
                    else:
                        return jsonify({"error": f"Point missing class or y value: {point}"}), 400
                except (ValueError, TypeError) as e:
                    return jsonify({"error": f"Invalid data value in point {point}: {e}"}), 400
                    
            # Convert to numpy arrays
            X_arr = np.array(X)
            y_arr = np.array(y)
            
            # Train the model
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42
            )
            clf.fit(X_arr, y_arr)
            
            # Generate tree visualization with only leaf nodes colored
            tree_img = generate_tree_visualization(clf, feature_names=['x1', 'x2'], class_names=np.unique(y_arr))
            
            # Generate optimized smaller decision boundary visualization
            boundary_img = generate_optimized_decision_boundary(clf, X_arr, y_arr)
            
            # Convert NumPy types to Python types (to ensure JSON serialization works)
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            result = {
                'tree_visualization': tree_img,
                'decision_boundary': boundary_img
            }
            
            return jsonify(convert_to_python_types(result))
            
        else:  # Regression case
            for point in training_data:
                try:
                    x1 = float(point['x1'])
                    x2 = float(point['x2'])
                    X.append([x1, x2])
                    
                    if 'value' in point:
                        y.append(float(point['value']))
                    elif 'y' in point:
                        y.append(float(point['y']))
                    else:
                        return jsonify({"error": f"Point missing value or y: {point}"}), 400
                except (ValueError, TypeError) as e:
                    return jsonify({"error": f"Invalid numeric value in point {point}: {e}"}), 400
                    
            # Convert to numpy arrays
            X_arr = np.array(X)
            y_arr = np.array(y)
            
            # Train the model
            regressor = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            regressor.fit(X_arr, y_arr)
            
            # Generate tree visualization 
            tree_img = generate_tree_visualization(regressor, feature_names=['x1', 'x2'], regression=True)
            
            # Generate regression surface visualization
            regression_img = generate_regression_surface(regressor, X_arr, y_arr)
            
            # Convert NumPy types to Python types
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            result = {
                'tree_visualization': tree_img,
                'decision_boundary': regression_img  # Use consistent key for frontend
            }
            
            return jsonify(convert_to_python_types(result))
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in dtree_visualize: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400


# Import the SVM module functions
from models.SVM import run_svm, generate_decision_boundary_only, predict_new_points

@app.route('/api/svm', methods=['POST', 'OPTIONS'])
def svm_endpoint():
    if request.method == 'OPTIONS':
        return handle_preflight_request()
    
    try:
        print("Received API request for SVM")
        data = request.get_json()
        
        print(f"Request data type: {type(data)}")
        print(f"Request data content: {data}")
        
        # Use the imported function from models/SVM.py
        result = run_svm(data)
        
        # Check if there was an error
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'traceback': result.get('traceback', '')
            }), 400
            
        # Return results
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/svm/sample_data', methods=['POST'])
def svm_sample_data_custom():
    try:
        data = request.json
        dataset_type = data.get('dataset_type', 'blobs')
        count = data.get('count', 40)
        n_clusters = data.get('n_clusters', 2)  # For SVM, default to 2 clusters
        variance = data.get('variance', 0.5)
        
        # Import the sample data generator from SVM module
        from models.SVM import generate_sample_data
        
        # Generate the sample data
        result = generate_sample_data(
            dataset_type=dataset_type,
            n_samples=count,
            n_clusters=n_clusters,
            variance=variance
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/ann/train', methods=['POST'])
def ann():
    try:
        data = request.json
        print(f"Received API request for ANN")
        print(f"Request data type: {type(data)}")
        print(f"Request data content: {data}")
        
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
        
        if 'X' not in data:
            return jsonify({"error": "Missing required data field: X must be provided"}), 400
        
        if not isinstance(data['X'], list):
            return jsonify({"error": "X must be an array/list"}), 400
        
        if len(data['X']) < 2:
            return jsonify({"error": "At least 2 data points are required for ANN"}), 400
            
        if 'y' not in data:
            return jsonify({"error": "Missing required data field: y (labels) must be provided"}), 400
            
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Run ANN algorithm
        result = run_ann(data)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in ANN endpoint: {str(e)}")
        print(error_traceback)
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

@app.route('/api/ann/predict', methods=['POST'])
def ann_predict():
    try:
        data = request.json
        print(f"Received API request for ANN prediction")
        
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
        
        if 'model' not in data:
            return jsonify({"error": "Missing required data field: model must be provided"}), 400
            
        if 'points' not in data:
            return jsonify({"error": "Missing required data field: points must be provided"}), 400
    
        # Extract the model from the data
        model = data['model']
        points = data['points']
        
        # Optional scaler if provided
        scaler = data.get('scaler', None)
        
        # Make predictions
        predictions = predict_points(model, points, scaler)
        
        return jsonify({"predictions": predictions})
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in ANN prediction endpoint: {str(e)}")
        print(error_traceback)
        return jsonify({"error": str(e), "traceback": error_traceback}), 500
    
@app.route('/api/ann/sample_data', methods=['POST'])
def ann_sample_data():
    try:
        data = request.json
        print(f"Received API request for ANN sample data")
        
        dataset_type = data.get('dataset_type', 'blobs')
        n_samples = data.get('count', 100)
        n_clusters = data.get('n_clusters', 2)
        variance = data.get('variance', 0.5)
        
        # Handle specific dataset types
        if (dataset_type == 'xor'):
            # XOR always has 2 classes
            n_clusters = 2
        elif dataset_type == 'circle':
            # Circle pattern uses 'circles' dataset type
            dataset_type = 'circles'
            n_clusters = 2  # circles dataset always has 2 classes
        elif dataset_type == 'spiral':
            # For spiral, n_clusters specifies the number of spiral arms
            n_clusters = max(2, min(n_clusters, 5))  # Limit between 2 and 5 spiral arms
        
        # Generate sample data
        sample_data = generate_sample_data(
            dataset_type=dataset_type,
            n_samples=n_samples,
            n_clusters=n_clusters,
            variance=variance
        )
        
        return jsonify(sample_data)
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in ANN sample data endpoint: {str(e)}")
        print(error_traceback)
        return jsonify({
            "error": str(e), 
            "traceback": error_traceback
        }), 500
    

from flask import jsonify, request
@app.route('/api/dbscan/run_complete', methods=['POST'])
def dbscan_run_complete():
    try:
        data = request.get_json()
        points = data.get('points', [])
        eps = float(data.get('eps', 0.5))
        min_samples = int(data.get('min_samples', 5))
        
        # Get visualization options - these will be used in the frontend, but we still pass them
        # to keep the API consistent
        show_core_points = data.get('show_core_points', True)
        show_border_points = data.get('show_border_points', True)
        show_noise_points = data.get('show_noise_points', True)
        show_epsilon_radius = data.get('show_epsilon_radius', True)
        
        if not points:
            return jsonify({"status": "error", "message": "No data points provided"}), 400
        
        from models.dbscan import run_dbscan
        result = run_dbscan(
            points=points, 
            eps=eps, 
            min_samples=min_samples,
            show_core_points=show_core_points,
            show_border_points=show_border_points,
            show_noise_points=show_noise_points,
            show_epsilon_radius=show_epsilon_radius
        )
        
        # Convert NumPy types to Python native types for JSON serialization
        def convert_to_python_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Convert the result before returning
        result = convert_to_python_types(result)
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in dbscan_run_complete: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 400

@app.route('/api/dbscan/sample_data', methods=['POST'])
def dbscan_sample_data():
    try:
        # Get parameters from request
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'blobs')
        n_samples = int(data.get('n_samples', 100))
        n_clusters = int(data.get('n_clusters', 3))
        noise_level = float(data.get('noise_level', 0.05))
        
        # Import the sample data generator
        from datasets.sample_data import generate_dbscan_data
        
        # Generate sample data
        result = generate_dbscan_data(
            dataset_type=dataset_type,
            n_samples=n_samples,
            n_clusters=n_clusters,
            noise_level=noise_level
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# For backwards compatibility, keep the GET endpoint too
@app.route('/api/dbscan/sample-data', methods=['GET'])
def get_dbscan_sample_data():
    try:
        # Get query parameters with defaults
        dataset_type = request.args.get('dataset_type', default='blobs', type=str)
        n_samples = request.args.get('n_samples', default=100, type=int)
        n_clusters = request.args.get('n_clusters', default=3, type=int)
        noise_level = request.args.get('noise_level', default=0.05, type=float)
        
        # Import the sample data generator
        from datasets.sample_data import generate_dbscan_data
        
        # Generate sample data
        result = generate_dbscan_data(
            dataset_type=dataset_type,
            n_samples=n_samples,
            n_clusters=n_clusters,
            noise_level=noise_level
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/regression', methods=['POST'])
def regression():
    """
    Endpoint for polynomial regression of any degree
    """
    data = request.json
    
    try:
        # Log incoming data for debugging
        print(f"Received API request for polynomial regression")
        print(f"Request data type: {type(data)}")
        print(f"Request data content: {data}")
        
        # Validate input data
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. Expected JSON object"}), 400
            
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "Missing required data fields: X and y must be provided"}), 400
            
        if not isinstance(data['X'], list) or not isinstance(data['y'], list):
            return jsonify({"error": "X and y must be arrays/lists"}), 400
            
        if len(data['X']) < 2:
            return jsonify({"error": "At least 2 data points are required for regression"}), 400
        
        if len(data['X']) != len(data['y']):
            return jsonify({"error": f"Length mismatch: X has {len(data['X'])} elements, y has {len(data['y'])} elements"}), 400
        
        # Extract parameters with defaults
        alpha = float(data.get('alpha', 0.01))
        iterations = int(data.get('iterations', 100))
        degree = int(data.get('degree', 1))  # Default to 1 for backward compatibility
        
        # Ensure iterations is an integer
        try:
            iterations = int(iterations)
            if iterations < 1:
                iterations = 100  # Default if invalid
        except (ValueError, TypeError):
            iterations = 100  # Default if invalid

        # Call the model with parameters
        print(f"Calling polynomial regression model function with degree={degree}, alpha={alpha}, iterations={iterations}")
        result = run_polynomial_regression(data, degree=degree, alpha=alpha, iterations=iterations)
        
        # Check if the result is None (which would cause JSON serialization issues)
        if result is None:
            return jsonify({"error": "Model returned None"}), 500
            
        # Check for error field    
        if isinstance(result, dict) and 'error' in result:
            print(f"Error in polynomial regression: {result['error']}")
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in polynomial regression endpoint: {str(e)}")
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

@app.route('/api/regression/sample_data', methods=['POST'])
def generate_regression_sample_data():
    """Generate sample data for polynomial regression visualization"""
    try:
        # Parse request parameters
        data = request.json
        dataset_type = data.get('dataset_type', 'linear')
        n_samples = min(int(data.get('n_samples', 30)), 100)  # Limit to 100 points
        noise_level = float(data.get('noise_level', 0.5))
        
        # Generate X values (evenly distributed in range)
        X = np.linspace(-8, 8, n_samples).reshape(-1, 1)
        
        # Generate y values based on dataset type
        if dataset_type == 'linear':
            # y = 2x + 1 + noise
            y = 2 * X.flatten() + 1 + np.random.normal(0, noise_level, n_samples)
            
        elif dataset_type == 'quadratic':
            # y = 0.5x² - 2x + 1 + noise
            y = 0.5 * X.flatten()**2 - 2 * X.flatten() +  + np.random.normal(0, noise_level, n_samples)
            
        elif dataset_type == 'sinusoidal':
            # y = 3sin(x) + noise
              y = 3 * np.sin(0.7 * X.flatten()) + np.random.normal(0, noise_level, n_samples)
            
        else:
            return jsonify({"error": f"Unknown dataset type: {dataset_type}"}), 400
        
        # Return data as JSON
        return jsonify({
            "X": X.flatten().tolist(),
            "y": y.tolist(),
            "dataset_type": dataset_type,
            "noise_level": noise_level
        })
        
    except Exception as e:
        print(f"Error generating sample data: {str(e)}")
        return jsonify({"error": f"Error generating sample data: {str(e)}"}), 500

if __name__ == '__main__':
  debug_mode = os.environ.get('FLASK_ENV') == 'development'
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)

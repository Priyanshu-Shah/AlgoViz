import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import io
import base64
import traceback

def predict_single_point(data, predict_point, n_neighbors=5):
    """
    Predict the class or value of a single point using KNN
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists of lists or numpy arrays
    predict_point (list): The point to predict [x1, x2]
    n_neighbors (int): Number of neighbors for KNN
    
    Returns:
    dict: Results including predicted class or value
    """
    try:
        # Convert to numpy arrays, ensuring they are float type for calculation
        X = np.array(data['X'], dtype=float)
        y_orig = np.array(data['y'])
        predict_point = np.array(predict_point, dtype=float).reshape(1, -1)
        
        # Validate inputs
        if X.shape[0] < 1:
            return {"error": "Need at least 1 training point for prediction"}
        
        if X.shape[1] != predict_point.shape[1]:
            return {"error": f"Prediction point has {predict_point.shape[1]} features, but training data has {X.shape[1]} features"}
        
        # Explicitly check for classification vs regression by examining unique y values
        unique_y = np.unique(y_orig)
        
        # If we only have a small number of unique values (like 2 for binary classification)
        # AND they are all close to integers, treat as classification
        if len(unique_y) <= 5 and all(abs(float(val) - round(float(val))) < 0.01 for val in unique_y):
            print("Using CLASSIFICATION mode")
            # For classification, convert all values to strings for consistency
            y = np.array([str(int(float(val))) for val in y_orig])
            
            # Train classification model with string classes
            model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            
            # Make prediction
            predicted_class = str(model.predict(predict_point)[0])
            
            print(f"Classification prediction: {predicted_class} (string)")
            
            return {
                'predicted_class': predicted_class
            }
        else:
            print("Using REGRESSION mode")
            # For regression, convert to float
            y = np.array([float(val) for val in y_orig])
            
            # Train regression model
            model = KNeighborsRegressor(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            
            # Make prediction
            predicted_value = model.predict(predict_point)[0]
            
            print(f"Regression prediction: {predicted_value} (float)")
            
            return {
                'predicted_class': str(round(predicted_value, 3))
            }
    
    except Exception as e:
        print(f"Error in predict_single_point: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": f"Error making prediction: {str(e)}"
        }

def generate_decision_boundary(data, n_neighbors=5):
    """
    Generate a decision boundary visualization for KNN
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists of lists or numpy arrays
    n_neighbors (int): Number of neighbors for KNN
    
    Returns:
    dict: Contains the decision boundary image as base64
    """
    try:
        # Convert to numpy arrays
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'])
        
        # Validate inputs
        if X.shape[0] < 5:
            return {"error": "Need at least 5 training points to generate a decision boundary"}
            
        if X.shape[1] != 2:
            return {"error": "Decision boundary visualization requires exactly 2 features"}
            
        # Check if we're doing classification or regression
        is_regression = False
        try:
            float_y = y.astype(float)
            is_regression = True
            # For regression, use a KNeighborsRegressor
            model = KNeighborsRegressor(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, float_y)
        except (ValueError, TypeError):
            # For classification, use a KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            
        # Create a plot of the decision boundary
        plt.figure(figsize=(8, 8))
            
        # Create a mesh grid
        h = 0.1  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Predict on the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Reshape the prediction results to match the grid
        Z = Z.reshape(xx.shape)
        
        if is_regression:
            # For regression, use a continuous colormap
            plt.contourf(xx, yy, Z, 50, cmap='viridis', alpha=0.5)
            plt.colorbar(label='Predicted Value')
            
            # Plot training points with color based on value
            sc = plt.scatter(X[:, 0], X[:, 1], c=float_y, cmap='viridis', 
                             edgecolor='k', s=80, alpha=0.7)
            
            plt.title(f'KNN Regression (k={n_neighbors})')
            
        else:
            # For classification, use discrete colors for classes
            classes = np.unique(y)
            cmap = plt.cm.coolwarm if len(classes) <= 2 else plt.cm.viridis
            
            plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
            
            # Plot training points with color based on class
            colors = list(mcolors.TABLEAU_COLORS)[:len(classes)]
            
            for i, cls in enumerate(classes):
                idx = np.where(y == cls)
                plt.scatter(X[idx, 0], X[idx, 1], c=[colors[i]], 
                            label=f'Class {cls}', edgecolor='k', s=80, alpha=0.7)
                
            plt.title(f'KNN Classification (k={n_neighbors})')
            plt.legend()
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        
        # Save the plot to a base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'decision_boundary': image_base64
        }
        
    except Exception as e:
        print(f"Error in generate_decision_boundary: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": f"Error generating decision boundary: {str(e)}"
        }
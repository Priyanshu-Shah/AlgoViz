import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib.colors as mcolors
import traceback

# Keep the existing classification and regression functions

# Updated prediction function without decision boundary
def predict_single_point(data, predict_point, n_neighbors=5):
    """
    Predict the class of a single point using KNN
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists of lists or numpy arrays
    predict_point (list): The point to predict [x1, x2]
    n_neighbors (int): Number of neighbors for KNN
    
    Returns:
    dict: Results including predicted class
    """
    try:
        # Convert to numpy arrays
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'])
        predict_point = np.array(predict_point, dtype=float).reshape(1, -1)
        
        # Validate inputs
        if X.shape[0] < 1:
            return {"error": "Need at least 1 training point for prediction"}
            
        if X.shape[1] != predict_point.shape[1]:
            return {"error": f"Prediction point has {predict_point.shape[1]} features, but training data has {X.shape[1]} features"}
        
        # Train the model
        model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
        model.fit(X, y)
        
        # Make prediction
        predicted_class = model.predict(predict_point)[0]
        
        return {
            'predicted_class': predicted_class
        }
    
    except Exception as e:
        return {
            "error": f"Error making prediction: {str(e)}"
        }
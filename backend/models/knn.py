import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import io
import base64
import traceback


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y[k_nearest_indices]
            
            # Return most common class
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
            
        return np.array(predictions)

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y, dtype=float)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get values of k nearest neighbors
            k_nearest_values = self.y[k_nearest_indices]
            
            # Return mean value
            predictions.append(np.mean(k_nearest_values))
            
        return np.array(predictions)

def predict_single_point(data, predict_point, n_neighbors=5):
    """
    Predict the class or value of a single point using KNN
    
    Parameters:
    data (dict): Contains 'X', 'y', and 'mode' ('classification' or 'regression')
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
        mode = data.get('mode', 'classification')  # Default to classification if not specified
        
        print(f"Predict point: {predict_point.tolist()[0]}")
        print(f"n_neighbors: {n_neighbors}")
        print(f"Mode: {mode}")
        
        # Validate inputs
        if X.shape[0] < 1:
            return {"error": "Need at least 1 training point for prediction"}
        
        if X.shape[1] != predict_point.shape[1]:
            return {"error": f"Prediction point has {predict_point.shape[1]} features, but training data has {X.shape[1]} features"}
        
        # Use explicit mode from frontend instead of trying to detect it
        if mode == 'classification':
            print("Using CLASSIFICATION mode")
            # For classification, strings are fine - ensure they're strings
            y = np.array([str(val) for val in y_orig])
            
            # Train classification model
            model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            
            # Make prediction
            predicted_class = str(model.predict(predict_point)[0])
            
            print(f"Classification prediction: {predicted_class}")
            
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
    data (dict): Contains 'X', 'y', and 'mode' ('classification' or 'regression')
    n_neighbors (int): Number of neighbors for KNN
    
    Returns:
    dict: Contains the decision boundary image as base64
    """
    try:
        # Convert to numpy arrays
        X = np.array(data['X'], dtype=float)
        y_orig = np.array(data['y'])
        mode = data.get('mode', 'classification')  # Default to classification if not specified
        
        print(f"Generating decision boundary with mode: {mode}")
        print(f"y sample: {y_orig[:5]}")
        
        # Validate inputs
        if X.shape[0] < 5:
            return {"error": "Need at least 5 training points to generate a decision boundary"}
            
        if X.shape[1] != 2:
            return {"error": "Decision boundary visualization requires exactly 2 features"}
        
        # Create a mesh grid
        h = 0.1  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Ensure we cover the -8 to 8 range to match frontend
        x_min = min(x_min, -8)
        x_max = max(x_max, 8)
        y_min = min(y_min, -8)
        y_max = max(y_max, 8)
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Create a plot of the decision boundary
        plt.figure(figsize=(8, 8))
        
        if mode == 'regression':
            print("Processing REGRESSION boundary")
            # For regression, ensure values are float
            y = np.array([float(val) for val in y_orig])
            
            model = KNeighborsRegressor(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            
            # Predict on the mesh grid
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # For regression, use a continuous colormap
            plt.contourf(xx, yy, Z, 50, cmap='viridis', alpha=0.8)
            plt.colorbar(label='Predicted Value')
            plt.title(f'KNN Regression Decision Regions (k={n_neighbors})')
            
        else:  # classification
            print("Processing CLASSIFICATION boundary")
            # For classification, ensure values are strings
            y = np.array([str(val) for val in y_orig])
            
            # Convert string classes to integers for contour plotting
            # This is the key change to fix the error
            unique_classes = np.unique(y)
            class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
            y_numeric = np.array([class_to_num[cls] for cls in y])
            
            model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)  # Fit with string classes
            
            # Predict on the mesh grid - get string predictions
            Z_strings = model.predict(np.c_[xx.ravel(), yy.ravel()])
            
            # Convert to numeric for plotting
            Z_numeric = np.array([class_to_num[cls] for cls in Z_strings])
            Z_numeric = Z_numeric.reshape(xx.shape)
            
            # Use a colormap that works well with 3 classes
            if len(unique_classes) <= 2:
                colors = ['#3b82f6', '#ef4444']
                cmap = mcolors.ListedColormap(colors[:len(unique_classes)])
            elif len(unique_classes) == 3:
                # Custom colormap for 3 classes: blue, red, green
                colors = ['#3b82f6', '#ef4444', '#22c55e']
                cmap = mcolors.ListedColormap(colors[:len(unique_classes)])
            else:
                cmap = plt.cm.tab10
            
            # Draw decision boundaries with SOLID COLORS using numeric Z values
            plt.contourf(xx, yy, Z_numeric, levels=len(unique_classes)-1, alpha=0.7, cmap=cmap)
            
            # Add thin black contour lines to show the exact boundaries between regions
            plt.contour(xx, yy, Z_numeric, levels=len(unique_classes)-1, colors='k', linewidths=0.5, alpha=0.5)
            
            # Add grid lines to match frontend
            plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
            
            # Add axis lines through the origin
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.title(f'KNN Classification Decision Boundaries (k={n_neighbors})')
            
            # Add a custom legend that matches the frontend colors
            from matplotlib.patches import Patch
            legend_elements = []
            color_map = {
                '1': '#3b82f6',  # Blue
                '2': '#ef4444',  # Red
                '3': '#22c55e'   # Green
            }
            
            # Create legend based on original class labels, not numeric values
            for cls in unique_classes:
                if cls in color_map:
                    color = color_map[cls]
                    legend_elements.append(Patch(facecolor=color, alpha=0.7, 
                                               label=f'Class {cls}'))
                else:
                    # Use index in unique_classes to determine color if not in map
                    idx = np.where(unique_classes == cls)[0][0]
                    if idx < len(colors):
                        color = colors[idx]
                        legend_elements.append(Patch(facecolor=color, alpha=0.7, 
                                              label=f'Class {cls}'))
            
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right')
        
        # Set axis limits to match frontend (-8 to 8)
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        
        # Add tick marks at regular intervals
        plt.xticks(range(-8, 9, 2))
        plt.yticks(range(-8, 9, 2))
        
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
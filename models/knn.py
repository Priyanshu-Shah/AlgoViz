import numpy as np
import pandas as pd
# Set matplotlib backend to non-interactive mode before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib.colors as mcolors

def run_knn_classification(data, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Run KNN classification on the provided data
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists of lists or numpy arrays
    n_neighbors (int): Number of neighbors for KNN
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Results including metrics and visualization
    """
    # Convert to numpy arrays
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate visualization (only works for 2D data)
    plt.figure(figsize=(10, 6))
    
    # Create a mesh grid for decision boundary visualization
    if X.shape[1] == 2:  # Only for 2D data
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Scale the mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(scaler.transform(grid_points))
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Plot training and test points
        unique_labels = np.unique(y)
        colors = list(mcolors.TABLEAU_COLORS)[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], 
                        color=colors[i], marker='o', alpha=0.7, label=f'Train (class {label})')
            plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1], 
                        color=colors[i], marker='^', alpha=1.0, label=f'Test (class {label})')
        
        plt.title(f'KNN Classification (k={n_neighbors})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Visualization only available for 2D data", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return results
    return {
        'accuracy': accuracy,
        'n_neighbors': n_neighbors,
        'plot': plot_data,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def run_knn_regression(data, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Run KNN regression on the provided data
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists or numpy arrays
    n_neighbors (int): Number of neighbors for KNN
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Results including metrics and visualization
    """
    # Convert to numpy arrays
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    
    if X.shape[1] == 1:
        # For 1D data we can visualize predictions vs actual values
        plt.scatter(X_train, y_train, color='blue', label='Training data')
        plt.scatter(X_test, y_test, color='green', label='Testing data')
        
        # Sort X values for line plot
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_line_scaled = scaler.transform(X_line)
        y_line = model.predict(X_line_scaled)
        plt.plot(X_line, y_line, color='red', linewidth=2, label='KNN Regression')
        
    else:
        # For higher dimensions, plot predicted vs actual
        plt.scatter(y_train, y_pred_train, color='blue', label='Training data')
        plt.scatter(y_test, y_pred_test, color='green', label='Testing data')
        
        # Plot identity line
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
    
    plt.title(f'KNN Regression (k={n_neighbors})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return results
    return {
        'n_neighbors': n_neighbors,
        'mse_train': mse_train,
        'r2_train': r2_train,
        'mse_test': mse_test,
        'r2_test': r2_test,
        'plot': plot_data
    }

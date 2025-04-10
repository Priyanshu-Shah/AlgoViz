import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import base64
from io import BytesIO

def run_svm(data):
    """
    Perform SVM on the input data and return results for decision boundary and support vectors.

    Parameters:
    data (dict): Input data containing:
        - X: List of data points (each point is a dict with 'x' and 'y').
        - y: List of class labels.
        - kernel: Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid').
        - gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        - degree: Degree for 'poly' kernel
        - coef0: Independent term in kernel function for 'poly' and 'sigmoid'
        - marginWidth: Width of margin visualization for linear kernels

    Returns:
    dict: Results containing:
        - decisionBoundary: Base64-encoded image of the decision boundary.
        - supportVectors: List of support vectors.
        - accuracy: Training accuracy of the SVM model.
        - model_info: Dictionary with model parameters and statistics.
    """
    try:
        # Extract data
        X = np.array([[point['x'], point['y']] for point in data['X']])
        y = np.array(data['y'])
        
        # Extract parameters
        kernel = data.get('kernel', 'linear')
        gamma = data.get('gamma', 'scale')
        degree = int(data.get('degree', 3))
        coef0 = float(data.get('coef0', 0.0))
        margin_width = float(data.get('marginWidth', 1.0))
        
        # Create and train the SVM model (without pipeline for simplicity)
        model = SVC(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=True,
            random_state=42
        )
        model.fit(X, y)
        
        # Get support vectors
        support_vectors = model.support_vectors_.tolist()
        
        # Get support vector indices and classes
        support_indices = model.support_
        support_classes = [int(y[i]) for i in support_indices]
        
        # Get number of support vectors per class
        n_support = model.n_support_.tolist()
        
        # Calculate accuracy
        accuracy = accuracy_score(y, model.predict(X))
        
        # Generate decision boundary visualization without data points
        decision_boundary = generate_decision_boundary_only(model, X, y, kernel, margin_width)
        
        # Get additional model info
        model_info = {
            'kernel': kernel,
            'gamma': gamma if gamma != 'scale' else 'scale',
            'degree': degree,
            'coef0': coef0,
            'marginWidth': margin_width,
            'n_support': n_support,
            'total_support_vectors': len(support_vectors),
            'classes': model.classes_.tolist(),
            'intercept': model.intercept_.tolist(),
            'supportVectorClasses': support_classes
        }
        
        # For 'linear' kernel, extract weight vectors
        if kernel == 'linear':
            model_info['weights'] = model.coef_.tolist()

        # Use camelCase for frontend consistency
        return {
            'decisionBoundary': decision_boundary,
            'supportVectors': support_vectors,
            'supportVectorClasses': support_classes,
            'accuracy': float(accuracy),
            'model_info': model_info
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_decision_boundary_only(model, X, y, kernel='linear', margin_width=1.0):
    """
    Generate a decision boundary plot for the SVM model without data points.

    Parameters:
    model: Trained SVM model.
    X: Training data points.
    y: Class labels.
    kernel: Kernel type used (for visualization adjustments)
    margin_width: Width factor for margin visualization (for all kernels)

    Returns:
    str: Base64-encoded image of the decision boundary.
    """
    try:
        # Set up the figure with a consistent size
        plt.figure(figsize=(6, 6), dpi=150)
        
        # Set limits aligned with the frontend scaling (-8 to 8)
        range_min, range_max = -8, 8
        
        # Create a fine mesh grid
        h = 0.05  # Step size
        xx, yy = np.meshgrid(np.arange(range_min, range_max, h), np.arange(range_min, range_max, h))
        
        # Get predictions
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Get decision function values (distance from the hyperplane)
        if hasattr(model, 'decision_function'):
            Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z_decision = Z_decision.reshape(xx.shape)
        
        # Create a custom colormap with higher saturation
        cmap_light = colors.LinearSegmentedColormap.from_list(
            'custom_cmap', [(0.3, 0.5, 0.95, 0.45), (0.95, 0.3, 0.3, 0.45)]
        )
        
        # Plot the decision boundary with better opacity
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.9)
        
        # Draw decision boundary and margins for any kernel type
        if hasattr(model, 'decision_function'):
            # Decision boundary is where decision function is 0
            plt.contour(xx, yy, Z_decision, colors='k', levels=[0], linewidths=1.2)
            
            # Margin lines are where decision function is +1 and -1
            # Scale the margins based on margin_width parameter
            margin_factor = 1.0 * margin_width
            plt.contour(xx, yy, Z_decision, colors='k', levels=[margin_factor], linestyles='dashed', linewidths=1.0, dashes=(5, 3))
            plt.contour(xx, yy, Z_decision, colors='k', levels=[-margin_factor], linestyles='dashed', linewidths=1.0, dashes=(5, 3))

        # Set the axes limits to match the frontend (-8 to 8 in both axes)
        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)
        
        # Remove axes labels and ticks since we'll overlay this on our canvas
        plt.axis('off')
        
        # Convert to image with transparency
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        image_png = buffer.getvalue()
        
        # Clean up plot
        plt.close()
        
        # Encode
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        return image_base64
    except Exception as e:
        import traceback
        print(f"Error generating decision boundary: {str(e)}")
        print(traceback.format_exc())
        return None

def predict_new_points(model, points):
    """
    Predict the class of new points using the trained SVM model.
    
    Parameters:
    model: Trained SVM model
    points: List of points to predict, each point is a dict with 'x' and 'y'
    
    Returns:
    Dictionary containing prediction results
    """
    try:
        # Convert points to numpy array
        X_new = np.array([[point['x'], point['y']] for point in points])
        
        # Use model to predict
        predictions = model.predict(X_new).tolist()
        
        # If model can return probabilities, include them
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new).tolist()
        
        # Format prediction results for the frontend
        result = []
        for i, point in enumerate(points):
            result.append({
                'x': point['x'],
                'y': point['y'],
                'class': predictions[i],
                'probability': probabilities[i] if probabilities else None
            })
            
        return {
            'predictions': result,
            'probabilities': probabilities
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_sample_data(dataset_type='blobs', n_samples=40, n_clusters=3, variance=0.5):
    """
    Generate sample classification data with different patterns for SVM
    
    Parameters:
    dataset_type (str): Type of dataset to generate ('blobs', 'moons', 'circles')
    n_samples (int): Number of samples per class
    n_clusters (int): Number of clusters (only for 'blobs')
    variance (float): Controls the spread/noise of the data
    
    Returns:
    dict: Dictionary containing X features and y labels
    """
    import numpy as np
    np.random.seed(42)
    
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    if dataset_type == 'moons':
        # Generate two interleaving half circles
        X, y = make_moons(n_samples=n_samples*2, noise=variance*0.1, random_state=42)
        # Scale to -8 to 8 range
        X = X * 4.5 - 2  # Scale from default (around 0-1) to -8 to 8 range
        # Convert to numbers for SVM (0 and 1)
        y = y.astype(int)
        
    elif dataset_type == 'circles':
        # Generate concentric circles
        X, y = make_circles(n_samples=n_samples*2, noise=variance*0.1, factor=0.5, random_state=42)
        # Scale to -8 to 8 range
        X = X * 6.5  # Scale from default (around -1 to 1) to -8 to 8
        # Convert to numbers for SVM (0 and 1)
        y = y.astype(int)
        
    else:  # 'blobs' (default)
        # Generate clusters
        centers = []
        # Create evenly spaced cluster centers
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            center_x = 4 * np.cos(angle)
            center_y = 4 * np.sin(angle)
            centers.append([center_x, center_y])
        
        X, y_numeric = make_blobs(
            n_samples=n_samples*n_clusters, 
            centers=centers,
            cluster_std=variance*1.5,
            random_state=42
        )
        
        # For SVM, ensure we only have binary classes (0 and 1)
        y = (y_numeric % 2).astype(int)
    
    # Convert to lists for JSON serialization
    X_list = X.tolist()
    y_list = y.tolist()
    
    return {
        'X': X_list,
        'y': y_list
    }
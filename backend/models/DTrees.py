import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import base64
from io import BytesIO, StringIO
import json
import pydot
import os

def run_decision_tree_classification(data, max_depth=3, min_samples_split=2, criterion='gini'):
    """
    Run decision tree classification on the provided data
    
    Parameters:
    data (dict): Contains X data points and y labels
    max_depth (int): Maximum depth of the tree
    min_samples_split (int): Minimum samples required to split a node
    criterion (str): The function to measure the quality of a split ('gini' or 'entropy')
    
    Returns:
    dict: Results including tree visualization, accuracy, prediction paths
    """
    try:
        # Convert data to numpy arrays
        X = np.array(data['X'])
        y = np.array(data['y']).astype(str)  # Convert to string for categorical labels
        
        # Train the model
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42
        )
        clf.fit(X, y)
        
        # Get accuracy
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Generate tree visualization
        tree_img = generate_tree_visualization(clf, feature_names=['x1', 'x2'], class_names=np.unique(y))
        
        # Generate decision boundary
        boundary_img = generate_decision_boundary(clf, X, y)
        
        # Get feature importances
        feature_importances = clf.feature_importances_.tolist()
        
        # Get prediction paths for future visualization
        decision_paths = []
        for i, sample in enumerate(X[:5]):  # Store paths for a few samples as example
            path = get_decision_path(clf, sample)
            decision_paths.append({
                'sample_idx': i,
                'path': path,
                'prediction': clf.predict([sample])[0]
            })
        
        return {
            'model_type': 'classification',
            'accuracy': float(accuracy),
            'n_nodes': clf.tree_.node_count,
            'n_leaves': clf.get_n_leaves(),
            'max_depth': clf.get_depth(),
            'feature_importances': feature_importances,
            'tree_visualization': tree_img,
            'decision_boundary': boundary_img,
            'example_paths': decision_paths
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
def generate_regression_surface(tree_model, X, y):
    """
    Generate visualization of decision surface for regression tree
    
    Parameters:
    tree_model: Trained decision tree regression model
    X: Training data features
    y: Target values
    
    Returns:
    str: Base64 encoded image of the regression surface
    """
    try:
        # Create a mesh grid with fewer points for efficiency
        h = 0.05  # Larger step size for faster rendering
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Predict on mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points).reshape(xx.shape)
        
        # Create figure
        plt.figure(figsize=(6, 4.5), dpi=80)
        
        # Plot the prediction surface using contourf with viridis colormap
        contour = plt.contourf(xx, yy, Z, 20, cmap='viridis', alpha=0.8)
        
        # Add scatter plot of training points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, 
                            cmap='viridis', edgecolor='k', s=30, alpha=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(contour, shrink=0.8)
        cbar.set_label('Predicted Value')
        
        # Add labels and title
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Regression Surface')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.tight_layout()
        
        # Convert to base64 image - REMOVE OPTIMIZE PARAMETER
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight',
                  pad_inches=0.1, transparent=False)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image
        
    except Exception as e:
        import traceback
        print(f"Error generating regression surface: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple error message image
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating regression surface:\n{str(e)}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image
        
def run_decision_tree_regression(data, max_depth=3, min_samples_split=2):
    """
    Run decision tree regression on the provided data
    """
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        # Check for valid input
        if len(X) < 2:
            return {"error": "Need at least 2 training points"}
            
        # Create and train the model
        regressor = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        regressor.fit(X, y)
        
        # Make predictions on training data
        y_pred = regressor.predict(X)
        
        # Calculate regression metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Generate tree visualization
        tree_img = generate_tree_visualization(regressor, feature_names=['x1', 'x2'], regression=True)
        
        # Generate regression surface visualization
        decision_boundary = generate_regression_surface(regressor, X, y)
        
        # Get feature importances
        feature_importances = regressor.feature_importances_.tolist()
        
        return {
            'model_type': 'regression',
            'mse': float(mse),
            'r2': float(r2),
            'n_nodes': regressor.tree_.node_count,
            'n_leaves': regressor.get_n_leaves(),
            'max_depth': regressor.get_depth(),
            'feature_importances': feature_importances,
            'tree_visualization': tree_img,
            'decision_boundary': decision_boundary  # Use the same key name for consistency
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def predict_data_points(data, predict_points, max_depth=3, min_samples_split=2, criterion='gini'):
    """
    Predict class labels for given data points
    """
    try:
        # Convert data to numpy arrays
        X = np.array(data['X'])
        y = np.array(data['y'])
        predict_X = np.array(predict_points)
        
        # Check if it's regression or classification based on y values
        is_regression = False
        try:
            # Try to convert y to float - if successful, it's regression
            float_y = np.array([float(val) for val in y])
            is_regression = True
            y = float_y  # Use float values for regression
        except (ValueError, TypeError):
            # If conversion fails, it's classification
            is_regression = False
        
        print(f"Prediction mode: {'regression' if is_regression else 'classification'}")
        print(f"Training data shape: {X.shape}, y shape: {y.shape}")
        print(f"Prediction data shape: {predict_X.shape}")
        
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
        
        # Predict labels for the new points
        predictions = model.predict(predict_X)
        
        # Add debugging for prediction values
        print(f"Raw predictions: {predictions}")
        print(f"Prediction type: {type(predictions)}, dtype: {predictions.dtype}")
        
        # For each prediction, get the decision path
        prediction_paths = []
        for i, point in enumerate(predict_X):
            path = get_decision_path(model, point)
            prediction_paths.append({
                'point_idx': i,
                'path': path,
                'prediction': str(predictions[i]) if not is_regression else float(predictions[i])
            })
        
        # Generate tree visualization
        if is_regression:
            tree_img = generate_tree_visualization(model, feature_names=['x1', 'x2'], regression=True)
            decision_boundary = generate_regression_surface(model, X, y)
        else:
            tree_img = generate_tree_visualization(model, feature_names=['x1', 'x2'], class_names=np.unique(y))
            decision_boundary = generate_optimized_decision_boundary(model, X, y)
        
        # Convert predictions to list for JSON serialization
        if is_regression:
            predictions_list = predictions.tolist()
        else:
            predictions_list = [str(p) for p in predictions]
            
        # Debug the formatted predictions
        print(f"Formatted predictions: {predictions_list}")
        
        # Add class mapping information to help frontend
        class_mapping = {}
        if not is_regression:
            unique_classes = np.unique(y)
            # Create a mapping from unique values to 0, 1, 2
            for i, cls in enumerate(unique_classes):
                class_mapping[i] = str(cls)
            print(f"Class mapping: {class_mapping}")
        
        return {
            'predictions': predictions_list,
            'prediction_paths': prediction_paths,
            'tree_visualization': tree_img,
            'decision_boundary': decision_boundary,
            'model_type': 'regression' if is_regression else 'classification',
            'class_mapping': class_mapping  # Add this to response
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_data_points: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def get_decision_path(tree_model, sample):
    """
    Get the decision path for a sample through the tree
    
    Parameters:
    tree_model: Trained decision tree model
    sample: Single data point
    
    Returns:
    list: Sequence of node indices representing the path
    """
    # Get the node indicator matrix
    node_indicator = tree_model.decision_path([sample])
    # Get the leaf id
    leaf_id = tree_model.apply([sample])[0]
    
    # Get the decision path for the sample
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # Collect the path information
    path_info = []
    for i, node_id in enumerate(node_index):
        if i < len(node_index) - 1:  # Not the last node
            # Get the threshold and feature for this node
            feature = tree_model.tree_.feature[node_id]
            threshold = tree_model.tree_.threshold[node_id]
            
            path_info.append({
                'node_id': int(node_id),
                'feature': int(feature),
                'threshold': float(threshold),
                'goes_left': sample[feature] <= threshold
            })
        else:  # Leaf node
            # Get prediction value at this node
            value = tree_model.tree_.value[node_id]
            path_info.append({
                'node_id': int(node_id),
                'is_leaf': True,
                'value': value.tolist()
            })
    
    return path_info

def generate_tree_visualization(model, feature_names=None, class_names=None, regression=False):
    """
    Generate a visualization of the decision tree with proper impurity values and only leaf nodes colored
    """
    import re  # Add the import at the function start
    import base64  # Make sure base64 is imported at the function level
    
    try:
        # Create a temporary file for dot output
        import tempfile
        dot_file = tempfile.NamedTemporaryFile(suffix='.dot', delete=False)
        dot_file_path = dot_file.name
        dot_file.close()
        
        # Export the decision tree to dot format with limited depth
        max_depth_to_show = min(5, model.get_depth())  # Limit visualization depth
        
        # Export to dot format with proper impurity values - Increase precision from 0 to 4
        export_graphviz(
            model, 
            out_file=dot_file_path,
            feature_names=feature_names,
            class_names=class_names,
            filled=False,  # No filling initially
            rounded=True,
            special_characters=True,
            max_depth=max_depth_to_show,
            proportion=False,
            precision=4,  # Increased precision for impurity values
            impurity=True  # Make sure impurity is shown
        )
        
        # Read the dot file content
        with open(dot_file_path, 'r') as f:
            dot_content = f.read()
        
        # Get the leaf nodes (nodes without children)
        leaf_nodes = set()
        # In the DOT format, leaf nodes don't have outgoing edges
        node_pattern = r'(\d+) \[label=<'
        edge_pattern = r'(\d+) -> (\d+)'
        
        # Find all nodes
        all_nodes = set()
        for match in re.finditer(node_pattern, dot_content):
            all_nodes.add(match.group(1))
        
        # Find nodes with outgoing edges
        non_leaf_nodes = set()
        for match in re.finditer(edge_pattern, dot_content):
            non_leaf_nodes.add(match.group(1))
        
        # Leaf nodes are nodes without outgoing edges
        leaf_nodes = all_nodes - non_leaf_nodes
        
        # First make all nodes transparent/white with light gray borders
        dot_content = dot_content.replace('node [shape=box]', 
                                         'node [shape=box, style=filled, fillcolor="white", color="#dddddd"]')
        
        # For non-leaf nodes, modify the label to remove class information but keep impurity
        if not regression:
            # Match pattern for non-leaf nodes
            non_leaf_pattern = r'(\d+) \[label=<(.*?)class = ([^<>\]]+)(.*?)\]'
            for node_id in non_leaf_nodes:
                # Find all matches
                for match in re.finditer(non_leaf_pattern, dot_content):
                    matched_node_id = match.group(1)
                    if matched_node_id == node_id:
                        before_class = match.group(2)
                        after_class = match.group(4)
                        # Remove class information for non-leaf nodes
                        new_label = f'{node_id} [label=<{before_class}{after_class}]'
                        dot_content = dot_content.replace(match.group(0), new_label)
        
        # Now color only the leaf nodes
        if not regression:
            # For classification trees, color leaf nodes by class
            leaf_pattern = r'(\d+) \[label=<.*?class = ([^<>\]]+).*?\]'
            for match in re.finditer(leaf_pattern, dot_content):
                node_id = match.group(1)
                
                # Only color if this is a leaf node
                if node_id in leaf_nodes:
                    class_name = match.group(2).strip()
                    
                    # Match exact colors from the frontend
                    class_colors = {
                        '0': '#3B82F6',  # Blue
                        '1': '#EF4444',  # Red
                        '2': '#22C55E',  # Green
                    }
                    
                    color = class_colors.get(class_name, '#AAAAAA')  # Default to gray if not found
                    
                    # Replace node style to add color
                    replace_pattern = f'{node_id} [label='
                    replacement = f'{node_id} [style=filled, fillcolor="{color}", color="black", label='
                    dot_content = dot_content.replace(replace_pattern, replacement)
        else:
            # For regression trees, apply gradient to leaf nodes only
            leaf_pattern = r'(\d+) \[label=<.*?value = \[([^\]]+)\].*?\]'
            for match in re.finditer(leaf_pattern, dot_content):
                node_id = match.group(1)
                
                # Only color if this is a leaf node
                if node_id in leaf_nodes:
                    value_str = match.group(2).strip()
                    
                    try:
                        # Get the predicted value from the node
                        parts = value_str.split()
                        if len(parts) > 0:
                            value = float(parts[0])
                            
                            # Normalize to 0-1 range for visualization
                            normalized_value = min(max(value / 10.0, 0), 1)
                            
                            # Use viridis-like color map: blue -> green -> yellow
                            if normalized_value < 0.5:
                                # Blue to green transition
                                ratio = normalized_value * 2
                                r = int(0 + ratio * 34)
                                g = int(119 + ratio * 78)
                                b = int(182 - ratio * 88)
                            else:
                                # Green to yellow transition
                                ratio = (normalized_value - 0.5) * 2
                                r = int(34 + ratio * 221)
                                g = int(197 - ratio * 17)
                                b = int(94 - ratio * 94)
                            
                            hex_color = f"#{r:02x}{g:02x}{b:02x}"
                            
                            # Replace node style to add color
                            replace_pattern = f'{node_id} [label='
                            replacement = f'{node_id} [style=filled, fillcolor="{hex_color}", color="black", label='
                            dot_content = dot_content.replace(replace_pattern, replacement)
                    except Exception as e:
                        print(f"Error processing regression value: {e}")
        
        # Add clickability for zooming - image map or JavaScript
        dot_content = dot_content.replace('digraph Tree {', 
                                        'digraph Tree {\n  graph [id=tree_viz, tooltip="Click to enlarge"]')
        
        # Write the modified dot content back to file
        with open(dot_file_path, 'w') as f:
            f.write(dot_content)
        
        # Convert dot file to PNG
        from subprocess import check_call
        png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        png_file_path = png_file.name
        png_file.close()
        
        # Use a smaller DPI setting for preview, but generate a higher-res version too
        check_call(['dot', '-Gdpi=72', '-Tpng', dot_file_path, '-o', png_file_path])
        
        # Read the PNG file
        with open(png_file_path, 'rb') as f:
            png_bytes = f.read()
            
        # Clean up temporary files
        import os
        os.unlink(dot_file_path)
        os.unlink(png_file_path)
        
        # Encode the PNG image as a base64 string
        base64_image = base64.b64encode(png_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{base64_image}"
        
    except Exception as e:
        import traceback
        import base64  # Make sure base64 is imported here too for exception handling
        from io import BytesIO
        import matplotlib.pyplot as plt
        
        print(f"Error generating tree visualization: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple error message image instead
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating tree visualization:\n{str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        
        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=72)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

def generate_decision_boundary(tree_model, X, y):
    """
    Generate a visualization of the decision boundary
    
    Parameters:
    tree_model: Trained decision tree model
    X: Training data features
    y: Training data labels
    
    Returns:
    str: Base64 encoded image of the decision boundary
    """
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    try:
        # Use explicit type conversion for the input data
        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points)
        
        # Ensure Z is numeric by converting to float if needed
        if not np.issubdtype(Z.dtype, np.number):
            # For classification, convert string/object to numbers based on unique values
            unique_z = np.unique(Z)
            z_map = {val: i for i, val in enumerate(unique_z)}
            Z = np.array([z_map[val] for val in Z], dtype=np.float64)
        
        Z = Z.reshape(xx.shape)
        
        # Create custom colormap for the unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Define colors for different classes (match frontend colors)
        colors = ['#22C55E', '#3B82F6', '#EF4444', '#F59E0B', '#8B5CF6']#EF4444
        # If more classes than colors, use default colormap
        if n_classes <= len(colors):
            cmap_light = ListedColormap(colors[:n_classes])
        else:
            cmap_light = plt.cm.rainbow
        
        # Plot the decision boundary with explicit type handling
        plt.figure(figsize=(8, 6))
        
        # Convert Z to float for plotting if it's not already
        Z_float = Z.astype(float)
        plt.contourf(xx, yy, Z_float, alpha=0.3, cmap=cmap_light)
        
        # Plot the training points
        for i, cls in enumerate(unique_classes):
            if i < len(colors):
                color = colors[i]
            else:
                color = plt.cm.rainbow(i / n_classes)
                
            idx = np.where(y == cls)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=f'Class {cls}', 
                       alpha=0.8, edgecolor='k')
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image
        
    except Exception as e:
        import traceback
        print(f"Error generating decision boundary: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple error message image instead
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error generating decision boundary:\n{str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image

def generate_regression_scatter_plot(X, y, predictions):
    """
    Generate a visualization of regression predictions
    
    Parameters:
    X: Training data features
    y: Training data labels (actual values)
    predictions: Model predictions for training data
    
    Returns:
    str: Base64 encoded image of the regression scatter plot
    """
    try:
        plt.figure(figsize=(6, 4.5), dpi=80)
        
        # Create a colormap for the actual values
        norm = plt.Normalize(min(y), max(y))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        
        # Plot the training data points colored by their actual values
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, 
                   alpha=0.8, edgecolor='k', s=30)
        
        # Create a mesh grid for predictions
        h = 0.1  # larger step size for faster rendering
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Predict over the grid
        Z = predictions.reshape(xx.shape)
        
        # Plot prediction contour
        contour = plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
        
        # Add colorbar
        plt.colorbar(sm, label='Value')
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Regression Predictions')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.tight_layout()
        
        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                  pad_inches=0.1, transparent=False, optimize=True)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image
        
    except Exception as e:
        import traceback
        print(f"Error generating regression visualization: {str(e)}")
        print(traceback.format_exc())
        
        # Create error image
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating regression visualization:\n{str(e)}", 
               horizontalalignment='center', verticalalignment='center',
               transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image

def generate_sample_classification_data(dataset_type='blobs', n_samples=40, n_clusters=3, variance=0.5):
    """
    Generate sample classification data with different patterns
    
    Parameters:
    dataset_type (str): Type of dataset to generate ('blobs', 'moons', 'circles')
    n_samples (int): Number of samples per class
    n_clusters (int): Number of clusters (only for 'blobs')
    variance (float): Controls the spread/noise of the data
    
    Returns:
    dict: Dictionary containing X features and y labels
    """
    np.random.seed(42)
    
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    if (dataset_type == 'moons'):
        # Generate two interleaving half circles
        X, y = make_moons(n_samples=n_samples*2, noise=variance*0.1, random_state=42)
        # Scale to -8 to 8 range
        X = X * 4.5 - 2 # Scale from default (around 0-1) to -8 to 8 range
        # Convert to strings to match expected format
        y = [str(int(i)) for i in y]
        
    elif (dataset_type == 'circles'):
        # Generate concentric circles
        X, y = make_circles(n_samples=n_samples*2, noise=variance*0.1, factor=0.5, random_state=42)
        # Scale to -8 to 8 range
        X = X * 6.5  # Scale from default (around -1 to 1) to -8 to 8
        # Convert to strings to match expected format
        y = [str(int(i)) for i in y]
        
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
        
        # Convert numeric y to string labels ('0', '1', '2', ...)
        y = [str(int(i)) for i in y_numeric]
    
    # Convert to lists for JSON serialization
    X_list = X.tolist()
    
    return {
        'X': X_list,
        'y': y
    }

def generate_sample_regression_data(dataset_type='nonlinear', n_samples=40, variance=0.5, sparsity=1.0):
    """
    Generate sample regression data with different patterns
    
    Parameters:
    dataset_type (str): Type of dataset to generate ('linear' or 'nonlinear')
    n_samples (int): Number of samples to generate
    variance (float): Controls the noise level
    sparsity (float): Controls how spread out the points are (1.0 = default, higher = more sparse)
    
    Returns:
    dict: Dictionary containing X features and y values
    """
    np.random.seed(42)
    
    # Generate random points in the -8 to 8 range
    # Use sparsity to control the distribution
    if sparsity > 1.0:
        # For higher sparsity, generate clusters of points instead of uniform distribution
        centers = []
        num_clusters = min(5, int(sparsity * 2))  # More sparsity = more clusters
        
        # Generate cluster centers
        for _ in range(num_clusters):
            center_x = np.random.uniform(-7, 7)
            center_y = np.random.uniform(-7, 7)
            centers.append([center_x, center_y])
        
        # Generate points around centers
        X = []
        points_per_cluster = n_samples // num_clusters
        remainder = n_samples % num_clusters
        
        for i, center in enumerate(centers):
            # Determine how many points to generate for this cluster
            cluster_points = points_per_cluster + (1 if i < remainder else 0)
            
            # Generate points with variance inversely proportional to sparsity
            cluster_variance = 2.0 / sparsity  # Smaller clusters for higher sparsity
            cluster_x = center[0] + np.random.normal(0, cluster_variance, cluster_points)
            cluster_y = center[1] + np.random.normal(0, cluster_variance, cluster_points)
            
            for j in range(cluster_points):
                X.append([cluster_x[j], cluster_y[j]])
        
        X = np.array(X)
    else:
        # For lower sparsity, use uniform distribution with adjusted range
        range_scale = 16.0 * sparsity  # Reduce range for lower sparsity
        X = np.random.uniform(-range_scale/2, range_scale/2, (n_samples, 2))
    
    # Generate y values based on the specified function type
    if dataset_type == 'linear':
        # Simple linear function: y = 0.5*x1 + 0.3*x2 + 2 + noise
        y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] 
        # Add noise scaled by variance
        y += np.random.normal(0, variance * 2.0, n_samples)
    else:  # nonlinear (default)
        # Nonlinear function with polynomial terms
        y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * (X[:, 0]**2) - 0.1 * (X[:, 0] * X[:, 1])
        # Add noise scaled by variance
        y += np.random.normal(0, variance * 2.0, n_samples)
    
    # Convert to lists for JSON serialization
    return {
        'X': X.tolist(),
        'y': y.tolist()
    }

def generate_tree_with_highlighted_path(tree_model, X, path_indices, feature_names=None, class_names=None, regression=False):
    """
    Generate a visualization of the decision tree with a highlighted path
    
    Parameters:
    tree_model: Trained decision tree model
    X: Sample point to trace through the tree
    path_indices: Indices of nodes in the decision path
    feature_names: List of feature names
    class_names: List of class names (for classification)
    regression: Boolean indicating if this is a regression tree
    
    Returns:
    str: Base64 encoded image of the tree visualization with highlighted path
    """
    # Create a temporary file for the graphviz output
    dotfile = StringIO()
    export_graphviz(
        tree_model,
        out_file=dotfile,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=not regression,
        precision=2
    )
    
    # Get the dot data and modify it to highlight the path
    dot_data = dotfile.getvalue()
    
    # Modify the dot data to highlight the path
    for idx in path_indices:
        # Add a red border to nodes in the path
        dot_data = dot_data.replace(f'node [shape=box, style="filled',
                                    f'node{idx} [shape=box, color=red, penwidth=3.0, style="filled')
    
    # Generate the tree image using pydot
    (graph,) = pydot.graph_from_dot_data(dot_data)
    
    # Create the PNG image
    png_bytes = graph.create_png()
    
    # Encode the PNG image as a base64 string
    base64_image = base64.b64encode(png_bytes).decode('utf-8')
    
    return base64_image

def generate_optimized_decision_boundary(tree_model, X, y):
    """
    Generate a smaller, optimized visualization of the decision boundary
    
    Parameters:
    tree_model: Trained decision tree model
    X: Training data features
    y: Training data labels
    
    Returns:
    str: Base64 encoded image of the decision boundary
    """
    try:
        # Create a mesh grid with fewer points
        h = 0.05  # Larger step size (0.05 instead of 0.02)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        Z = tree_model.predict(mesh_points)
        
        # Ensure Z is numeric
        if not np.issubdtype(Z.dtype, np.number):
            unique_z = np.unique(Z)
            z_map = {val: i for i, val in enumerate(unique_z)}
            Z = np.array([z_map[val] for val in Z], dtype=np.float64)
        
        Z = Z.reshape(xx.shape)
        
        # Create custom colormap
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Define colors for different classes (match frontend colors)
        colors = ['#3B82F6', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6']
        if n_classes <= len(colors):
            cmap_light = ListedColormap(colors[:n_classes])
        else:
            cmap_light = plt.cm.rainbow
        
        # Create a smaller figure
        plt.figure(figsize=(6, 4.5), dpi=80)
        
        # Plot decision boundary with reduced alpha for better visibility
        plt.contourf(xx, yy, Z.astype(float), alpha=0.3, cmap=cmap_light)
        
        # Plot data points with smaller markers
        for i, cls in enumerate(unique_classes):
            if i < len(colors):
                color = colors[i]
            else:
                color = plt.cm.rainbow(i / n_classes)
                
            idx = np.where(y == cls)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=f'Class {cls}', 
                       alpha=0.8, edgecolor='k', s=30)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        
        # Convert plot to base64 string with compression
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                    pad_inches=0.1, transparent=False, optimize=True)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image
        
    except Exception as e:
        import traceback
        print(f"Error generating optimized decision boundary: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple error message image
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Error generating boundary:\n{str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()
        buffer.seek(0)
        
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_image


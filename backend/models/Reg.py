import numpy as np
import pandas as pd
# Set matplotlib backend to non-interactive mode before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import io
import base64
import traceback  # Add this line to import traceback module

def gradient_descent(X, y, learning_rate=0.01, n_iterations=100, tolerance=1e-6, polynomial_degree=1, 
                   record_steps=False, step_interval=10):
    """
    Implements gradient descent for polynomial regression with safeguards for high learning rates
    
    Parameters:
    X (numpy.ndarray): Feature matrix (can be polynomial features)
    y (numpy.ndarray): Target vector
    learning_rate (float): Learning rate for gradient descent
    n_iterations (int): Maximum number of iterations
    tolerance (float): Convergence threshold for stopping criterion
    polynomial_degree (int): Degree of polynomial to adjust learning rate
    record_steps (bool): Whether to record theta at intervals for visualization
    step_interval (int): Number of iterations between recording history
    
    Returns:
    tuple: (coefficients, intercept, cost_history, actual_iterations, theta_history)
    """
    # Add a column of ones to X for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Modified learning rate scaling - better balance for higher degrees
    # Use a less aggressive scaling for polynomial degree
    if polynomial_degree <= 3:
        scaled_learning_rate = learning_rate
    elif polynomial_degree <= 5:
        scaled_learning_rate = learning_rate / (1 + 0.2 * (polynomial_degree - 3))
    else:
        scaled_learning_rate = learning_rate / (1 + 0.4 * (polynomial_degree - 5))
        
    print(f"Original learning rate: {learning_rate}, Scaled learning rate: {scaled_learning_rate}")
    
    # Initialize parameters (theta) with small random values to help break symmetry
    # This helps with fitting certain patterns like sinusoidal curves
    theta = np.random.randn(X_b.shape[1]) * 0.01
    
    # Initialize cost history and theta history
    cost_history = []
    theta_history = [] if record_steps else None
    
    # Feature scaling for better convergence
    # Scale both X and y for better numerical stability
    if X_b.shape[1] > 1:
        feature_means = np.mean(X_b[:, 1:], axis=0)
        feature_stds = np.std(X_b[:, 1:], axis=0)
        # Prevent division by zero
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        
        # Apply scaling to features (not to the intercept column)
        X_b_scaled = X_b.copy()
        X_b_scaled[:, 1:] = (X_b[:, 1:] - feature_means) / feature_stds
        
        # For sinusoidal-like data (which might have higher magnitude),
        # we also normalize the target values
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) > 0 else 1
        y_scaled = (y - y_mean) / y_std
    else:
        X_b_scaled = X_b
        feature_means = np.array([])
        feature_stds = np.array([])
        y_scaled = y
        y_mean = 0
        y_std = 1
    
    # Rest of the function remains the same, but we use y_scaled instead of y
    actual_iterations = 0
    prev_cost = float('inf')
    
    for i in range(n_iterations):
        actual_iterations += 1
        
        # Calculate predictions
        y_pred = X_b_scaled.dot(theta)
        
        # Calculate error using scaled y
        error = y_pred - y_scaled
        
        # Calculate cost (MSE)
        cost = np.mean(error**2)
        
        # Check for NaN or infinite values
        if np.isnan(cost) or np.isinf(cost) or cost > 1e10:
            print(f"Warning: Divergence detected at iteration {i}. Learning rate too high.")
            # If we've already completed some iterations and have a cost history, 
            # we can use the last good theta values
            if i > 5 and len(cost_history) > 5:
                break
            else:
                # Try reducing learning rate and restarting
                return gradient_descent(X, y, learning_rate=learning_rate*0.1, n_iterations=n_iterations, 
                                      tolerance=tolerance, polynomial_degree=polynomial_degree,
                                      record_steps=record_steps, step_interval=step_interval)
        
        # Check for divergence (cost increasing significantly)
        if i > 0 and cost > prev_cost * 1.5 and i > 5:
            print(f"Warning: Cost increasing at iteration {i}. Reducing learning rate.")
            # Try again with a smaller learning rate
            return gradient_descent(X, y, learning_rate=scaled_learning_rate*0.1, n_iterations=n_iterations, 
                                  tolerance=tolerance, polynomial_degree=polynomial_degree,
                                  record_steps=record_steps, step_interval=step_interval)
            
        cost_history.append(float(cost))
        prev_cost = cost
        
        # Record theta history at specified intervals if requested
        if record_steps and i % step_interval == 0:
            # Convert theta to original scale before recording
            if X_b.shape[1] > 1:
                # Extract intercept and coefficients
                intercept = theta[0] * y_std + y_mean
                coefficients = theta[1:] / feature_stds * y_std
                
                # Adjust intercept to account for mean-centered features
                intercept = intercept - np.sum(coefficients * feature_means)
                
                # Combine back for history recording
                orig_theta = np.concatenate([[intercept], coefficients])
            else:
                orig_theta = theta.copy() * y_std + y_mean
                
            theta_history.append(orig_theta)
        
        # Calculate gradients
        gradients = 2/X_b_scaled.shape[0] * X_b_scaled.T.dot(error)
        
        # Adaptive gradient clipping for higher degree polynomials
        if polynomial_degree > 3:
            # More aggressive clipping for higher degrees
            max_grad = 5.0 / polynomial_degree
            gradients = np.clip(gradients, -max_grad, max_grad)
        
        # Update parameters with adaptive learning rate
        theta = theta - scaled_learning_rate * gradients
        
        # Check for convergence
        if i > 0 and np.abs(cost_history[i] - cost_history[i-1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            
            # Record final theta if we're tracking history
            if record_steps and (i % step_interval != 0):
                if X_b.shape[1] > 1:
                    intercept = theta[0] * y_std + y_mean
                    coefficients = theta[1:] / feature_stds * y_std
                    intercept = intercept - np.sum(coefficients * feature_means)
                    orig_theta = np.concatenate([[intercept], coefficients])
                else:
                    orig_theta = theta.copy() * y_std + y_mean
                theta_history.append(orig_theta)
                
            break
    
    # Extract intercept and coefficients and transform back to original scale
    if X_b.shape[1] > 1:
        # For each coefficient, apply the inverse scaling
        coefficients = theta[1:] / feature_stds * y_std
        
        # Adjust intercept to account for mean-centered features and y scaling
        intercept = theta[0] * y_std + y_mean - np.sum(coefficients * feature_means)
    else:
        coefficients = theta[1:] * y_std
        intercept = theta[0] * y_std + y_mean
    
    return coefficients, intercept, cost_history, actual_iterations, theta_history

def run_polynomial_regression(data, degree=1, alpha=0.01, iterations=100, random_state=42, record_steps=False, step_interval=10):
    """
    Runs polynomial regression of specified degree on the provided data
    """
    try:
        # Set random seed
        np.random.seed(random_state)
        
        # Debug information
        print(f"Received data: X={data['X'][:5]}... (length: {len(data['X'])}), y={data['y'][:5]}... (length: {len(data['y'])})")
        print(f"Polynomial degree: {degree}, Learning rate (alpha): {alpha}, Max iterations: {iterations}")
        print(f"Record steps: {record_steps}, Step interval: {step_interval}")
        
        # Convert to numpy arrays with validation
        try:
            X = np.array(data['X'], dtype=float).reshape(-1, 1)
            y = np.array(data['y'], dtype=float)
            
            # Log the data shape
            print(f"Converted to numpy arrays: X shape {X.shape}, y shape {y.shape}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting data to numeric format: {str(e)}. Please ensure all values are valid numbers.")
        
        # Verify data integrity
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN values. Please check your input.")
        
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} elements, y has {len(y)} elements.")
        
        if len(X) < 2:
            raise ValueError("At least 2 data points are required for regression.")
        
        # Add a small amount of noise if all X values are the same
        if np.all(X == X[0]):
            print("Warning: All X values are identical. Adding small noise for numerical stability.")
            X = X + np.random.normal(0, 0.01, X.shape)
        
        # Transform features to polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Always enable recording steps 
        record_steps = True
        # Use step interval 1 for every iteration, will adjust later if needed
        effective_step_interval = 1
        
        # Train the model using gradient descent with history tracking
        coefficients, intercept, cost_history, actual_iterations, theta_history = gradient_descent(
            X_poly, y, learning_rate=alpha, n_iterations=iterations + 1, polynomial_degree=degree,
            record_steps=record_steps, step_interval=effective_step_interval
        )
        
        # Create model predictions for original data points
        X_poly_pred = poly.transform(X)
        y_pred = intercept + np.dot(X_poly_pred, coefficients)
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        mse = float(mean_squared_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        
        # Generate visualization of cost history
        cost_history_plot = create_cost_history_plot(cost_history)
        
        # Create iteration history for frontend visualization
        iteration_history = []
        
        # Debug the theta history
        print(f"Theta history length: {len(theta_history) if theta_history else 0}")
        print(f"Actual iterations completed: {actual_iterations}")
        
        if theta_history:
            # Determine if we need to subsample for the front-end
            # If there are more than 100 iterations, we'll subsample
            if len(theta_history) > 100:
                # Calculate step interval to get approximately 100 points
                front_end_step = max(1, len(theta_history) // 100)
                print(f"Many iterations ({len(theta_history)}), using step interval of {front_end_step} for frontend")
                selected_indices = range(0, len(theta_history), front_end_step)
                selected_theta_history = [theta_history[i] for i in selected_indices]
            else:
                # If fewer than 100 iterations, use all of them
                print(f"Few iterations ({len(theta_history)}), using all for frontend")
                selected_indices = range(len(theta_history))
                selected_theta_history = theta_history
            
            # Create the iteration history for each selected theta
            for i, idx in enumerate(selected_indices):
                theta = theta_history[idx]
                iter_intercept = theta[0]
                iter_coeffs = theta[1:]
                
                # Use the actual iteration number
                iter_step = idx
                iter_cost = cost_history[iter_step] if iter_step < len(cost_history) else cost_history[-1]
                
                iteration_history.append({
                    "iteration": iter_step,
                    "coefficients": iter_coeffs.tolist(),
                    "intercept": float(iter_intercept),
                    "cost": float(iter_cost),
                    "degree": int(degree)
                })
            
            print(f"Created iteration history with {len(iteration_history)} entries")
        else:
            print("Warning: No theta history was recorded, iteration playback will not be available.")
        
        # Return results
        results = {
            "coefficients": coefficients.tolist(),
            "intercept": float(intercept),
            "mse": mse,
            "r2": r2,
            "cost_history": cost_history,
            "cost_history_plot": cost_history_plot,
            "degree": int(degree),
            "alpha": float(alpha),
            "iterations": int(actual_iterations),
            "iteration_history": iteration_history
        }
        
        return results
        
    except Exception as e:
        print(f"Error in Polynomial Regression model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return error message
        error_msg = f"{str(e)}"
        if "learning rate" in str(e).lower() or "diverg" in str(e).lower():
            error_msg = f"Learning rate too high: {str(e)}"
        elif "nan" in str(e).lower() or "inf" in str(e).lower():
            error_msg = f"Numerical error (try reducing learning rate): {str(e)}"
        else:
            error_msg = f"{str(e)} - Unknown error"
        
        raise ValueError(f"Error in polynomial regression: {error_msg}")

# Add this function before run_polynomial_regression
def create_cost_history_plot(cost_history):
    """
    Creates a visualization of the cost history during training
    
    Parameters:
    cost_history (list): List of cost values at each iteration
    
    Returns:
    str: Base64 encoded PNG image of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cost history
    ax.plot(range(len(cost_history)), cost_history, color='blue', linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Iteration', fontsize=12, labelpad=10)
    ax.set_ylabel('Mean Squared Error', fontsize=12, labelpad=10)
    ax.set_title('Cost History during Training', fontsize=14, pad=20)
    ax.grid(alpha=0.3)
    
    # Use log scale for y-axis if there's a large range in cost values
    if max(cost_history) > 10 * min(cost_history):
        ax.set_yscale('log')
    
    plt.tight_layout(pad=2.0)
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return plot_base64

def create_regression_plot(X, y, coefficients, intercept, degree):
    """
    Creates a visualization of the polynomial regression fit
    
    Parameters:
    X (numpy.ndarray): Input features
    y (numpy.ndarray): Target values
    coefficients (numpy.ndarray): Polynomial coefficients
    intercept (float): Intercept term
    degree (int): Polynomial degree
    
    Returns:
    str: Base64 encoded PNG image of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(X, y, color='blue', alpha=0.6, label='Data Points')
    
    # Create polynomial features transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Generate smooth curve for plotting
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_poly = poly.fit_transform(X_range)
    y_pred = intercept + np.dot(X_poly, coefficients)
    
    # Plot regression line/curve
    ax.plot(X_range, y_pred, color='red', linewidth=2, label=f'Polynomial (Degree {degree})')
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('y', fontsize=12, labelpad=10)
    ax.set_title(f'Polynomial Regression (Degree {degree})', fontsize=14, pad=20)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Add padding to the plot (20% on each side)
    x_range = max(X) - min(X)
    y_range = max(y) - min(y)
    
    ax.set_xlim(min(X) - 0.2 * x_range, max(X) + 0.2 * x_range)
    ax.set_ylim(min(y) - 0.2 * y_range, max(y) + 0.2 * y_range)
    
    plt.tight_layout(pad=2.0)
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return plot_base64
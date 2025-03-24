import numpy as np
import pandas as pd
# Set matplotlib backend to non-interactive mode before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

def gradient_descent(X, y, learning_rate=0.01, n_iterations=100, tolerance=1e-6):
    """
    Implements gradient descent for linear regression with safeguards for high learning rates
    
    Parameters:
    X (numpy.ndarray): Feature matrix
    y (numpy.ndarray): Target vector
    learning_rate (float): Learning rate for gradient descent
    n_iterations (int): Maximum number of iterations
    tolerance (float): Convergence threshold for stopping criterion
    
    Returns:
    tuple: (coefficient, intercept, cost_history, actual_iterations)
    """
    # Add a column of ones to X for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize parameters (theta)
    theta = np.zeros(X_b.shape[1])  # Start with zeros instead of random values
    
    # Initialize cost history
    cost_history = []
    
    # Gradient descent
    actual_iterations = 0
    for i in range(n_iterations):
        actual_iterations += 1
        # Calculate predictions
        y_pred = X_b.dot(theta)
        
        # Calculate error
        error = y_pred - y
        
        # Calculate cost (MSE)
        cost = np.mean(error**2)
        
        # Check for NaN or infinite values
        if np.isnan(cost) or np.isinf(cost) or cost > 1e10:
            print(f"Warning: Divergence detected at iteration {i}. Learning rate too high.")
            break
            
        cost_history.append(float(cost))
        
        # Calculate gradients
        gradients = 2/X_b.shape[0] * X_b.T.dot(error)
        
        # Clip gradients to prevent extreme values
        max_grad = 10.0
        gradients = np.clip(gradients, -max_grad, max_grad)
        
        # Store old theta for adaptive learning rate
        theta_old = theta.copy()
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Check for convergence
        if i > 0 and np.abs(cost_history[i] - cost_history[i-1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
    
    # Extract intercept and coefficient
    intercept = theta[0]
    coefficient = theta[1]
    
    return coefficient, intercept, cost_history, actual_iterations

def run_linear_regression(data, alpha=0.01, iterations=100, random_state=42):
    """
    Run linear regression on the provided data using gradient descent
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists or numpy arrays
    alpha (float): Learning rate for gradient descent
    iterations (int): Maximum number of iterations for gradient descent
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Results including model coefficients, metrics, and visualization
    """
    try:
        # Set random seed
        np.random.seed(random_state)
        
        # Debug information
        print(f"Received data: X={data['X'][:5]}... (length: {len(data['X'])}), y={data['y'][:5]}... (length: {len(data['y'])})")
        print(f"Learning rate (alpha): {alpha}, Max iterations: {iterations}")
        
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
        
        # Train the model using gradient descent
        coef, intercept, cost_history, actual_iterations = gradient_descent(
            X, y, learning_rate=alpha, n_iterations=iterations + 1
        )
        
        # Create model predictions
        y_pred = coef * X.flatten() + intercept
        
        # Calculate metrics
        mse = float(mean_squared_error(y, y_pred))
        r2 = float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(X, y, color='blue', label='Data points')
        
        # Sort X values for line plot
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = coef * X_line.flatten() + intercept
        
        # Plot regression line
        plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
        
        # Add title and labels
        plt.title(f"Linear Regression with Gradient Descent (α={alpha:.4f}, iterations={actual_iterations})")
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Debug info
        print(f"Model trained successfully: coef={coef}, intercept={intercept}")
        print(f"Metrics: R2={r2}, MSE={mse}")
        print(f"Iterations used: {actual_iterations} out of {iterations} max")
        
        # Return results with explicit type conversion for all values
        result = {
            'coefficient': float(coef),
            'intercept': float(intercept),
            'mse': float(mse),
            'r2': float(r2),
            'alpha': float(alpha),
            'iterations': int(actual_iterations),
            'max_iterations': int(iterations),
            'final_cost': float(cost_history[-1]) if cost_history else 0.0,
            'plot': plot_data,
            'equation': f'y = {coef:.4f}x + {intercept:.4f}'
        }
        
        # Additional validation to ensure all values are properly serializable
        for key in result:
            if key not in ['plot', 'equation']:
                try:
                    if key in ['iterations', 'max_iterations']:
                        result[key] = int(result[key])
                    else:
                        result[key] = float(result[key])
                except (ValueError, TypeError):
                    print(f"Warning: Value for {key} could not be converted. Setting to default.")
                    if key in ['iterations', 'max_iterations']:
                        result[key] = 0
                    else:
                        result[key] = 0.0
        
        # Verify the result structure before returning
        print(f"Returning result keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        # Provide detailed error information
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in Linear Regression model: {str(e)}")
        print(f"Traceback: {error_traceback}")
        
        # Provide specific diagnosis for common issues
        error_diagnosis = "Unknown error"
        if "shape" in str(e).lower():
            error_diagnosis = "Data shape issue: Make sure X and y have compatible dimensions"
        elif "nan" in str(e).lower() or "infinity" in str(e).lower():
            error_diagnosis = "Data contains NaN or infinite values"
        elif "conversion" in str(e).lower() or "convert" in str(e).lower():
            error_diagnosis = "Data type conversion error: All values must be numeric"
        
        return {
            'error': f"{str(e)} - {error_diagnosis}",
            'traceback': error_traceback,
            'data_sample': {
                'X_sample': str(data.get('X', [])[:5]) if isinstance(data.get('X', None), list) else "Invalid X data",
                'y_sample': str(data.get('y', [])[:5]) if isinstance(data.get('y', None), list) else "Invalid y data"
            }
        }
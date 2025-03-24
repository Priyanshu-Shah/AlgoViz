import numpy as np
import pandas as pd
# Set matplotlib backend to non-interactive mode before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import io
import base64

def run_linear_regression(data, test_size=0.2, random_state=42):
    """
    Run linear regression on the provided data
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists or numpy arrays
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Results including model coefficients, metrics, and visualization
    """
    try:
        # Debug information
        print(f"Received data: X={data['X'][:5]}... (length: {len(data['X'])}), y={data['y'][:5]}... (length: {len(data['y'])})")
        
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
        
        if len(X) < 3:
            raise ValueError("At least 3 data points are required for regression.")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        mse_train = float(mean_squared_error(y_train, y_pred_train))
        r2_train = float(r2_score(y_train, y_pred_train))
        mse_test = float(mean_squared_error(y_test, y_pred_test))
        r2_test = float(r2_score(y_test, y_pred_test))
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training data')
        plt.scatter(X_test, y_test, color='green', label='Testing data')
        
        # Sort X values for line plot
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
        
        plt.title('Linear Regression Model')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        coef = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # Debug info
        print(f"Model trained successfully: coef={coef}, intercept={intercept}")
        print(f"Metrics: R2 train={r2_train}, R2 test={r2_test}")
        
        # Return results with explicit type conversion for all values
        result = {
            'coefficient': float(coef),
            'intercept': float(intercept),
            'mse_train': float(mse_train),
            'r2_train': float(r2_train),
            'mse_test': float(mse_test),
            'r2_test': float(r2_test),
            'plot': plot_data,
            'equation': f'y = {coef:.4f}x + {intercept:.4f}'
        }
        
        # Additional validation to ensure all values are properly serializable
        for key in result:
            if key not in ['plot', 'equation']:
                try:
                    result[key] = float(result[key])
                except (ValueError, TypeError):
                    print(f"Warning: Value for {key} could not be converted to float. Setting to 0.")
                    result[key] = 0.0
        
        # Verify the result structure before returning
        print(f"Returning result keys: {list(result.keys())}")
        print(f"Result data types: {[(k, type(v)) for k, v in result.items() if k != 'plot']}")
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

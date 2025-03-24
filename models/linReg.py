import numpy as np
import pandas as pd
# Set matplotlib backend to non-interactive mode before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

def run_linear_regression(data, random_state=42):
    """
    Run linear regression on the provided data without train/test split
    
    Parameters:
    data (dict): Contains 'X' and 'y' as lists or numpy arrays
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
        
        if len(X) < 2:
            raise ValueError("At least 2 data points are required for regression.")
        
        # Train the model on all data
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = float(mean_squared_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        
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
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        coef = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # Debug info
        print(f"Model trained successfully: coef={coef}, intercept={intercept}")
        print(f"Metrics: R2={r2}, MSE={mse}")
        
        # Return results with explicit type conversion for all values
        result = {
            'coefficient': float(coef),
            'intercept': float(intercept),
            'mse': float(mse),
            'r2': float(r2),
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
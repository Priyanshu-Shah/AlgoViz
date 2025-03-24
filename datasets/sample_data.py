import numpy as np

def generate_linear_data(n_samples=100, noise=10.0, seed=42):
    """
    Generate sample data for linear regression with controlled noise
    
    Parameters:
    n_samples (int): Number of samples to generate
    noise (float): Amount of noise to add
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary with 'X' and 'y' keys containing the data
    """
    np.random.seed(seed)
    
    # Generate X values
    X = np.linspace(0, 10, n_samples)
    
    # True relationship: y = 2x + 5 + noise
    true_coef = 2.0
    true_intercept = 5.0
    
    # Generate y values with noise
    y = true_coef * X + true_intercept + np.random.normal(0, noise, n_samples)
    
    return {
        'X': X.tolist(),
        'y': y.tolist(),
        'true_coef': true_coef,
        'true_intercept': true_intercept
    }

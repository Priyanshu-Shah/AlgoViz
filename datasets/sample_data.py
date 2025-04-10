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
    X = np.linspace(-8, 8, n_samples)
    
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


def generate_pca_data(n_samples=50, seed=42):
    """
    Generate sample data for PCA visualization
    
    Parameters:
    n_samples (int): Number of samples to generate
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary with 'X' key containing the 2D data points
    """
    import numpy as np
    np.random.seed(seed)
    
    # Generate a correlated dataset
    # We'll create data with correlation between x and y
    mean = [0, 0]
    
    # Covariance matrix - create a correlation of 0.8 between variables
    cov = [[1.0, 0.8], [0.8, 1.0]]
    
    # Generate random data from multivariate normal distribution
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    return {
        'X': data.tolist()
    }

def generate_dbscan_data(dataset_type='blobs', n_samples=100, n_clusters=3, noise_level=0.05):
    """
    Generate sample data for DBSCAN visualization
    
    Parameters:
    dataset_type (str): Type of dataset to generate ('blobs', 'moons', 'circles', 'anisotropic', 'noisy_circles')
    n_samples (int): Number of samples
    n_clusters (int): Number of clusters (only for 'blobs')
    noise_level (float): Controls the amount of noise in the dataset
    
    Returns:
    dict: Dictionary containing points as list of dicts with x,y coordinates
    """
    import numpy as np
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    np.random.seed(42)
    
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
        # Scale to better fill the canvas
        X = X * 5 - 2.5
        
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)
        # Scale to better fill the canvas
        X = X * 5
        
    elif dataset_type == 'anisotropic':
        # Create anisotropically distributed data
        X, _ = make_blobs(n_samples=n_samples, centers=1, random_state=42)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation) * 4
        
    elif dataset_type == 'noisy_circles':
        X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=noise_level*2, random_state=42)
        # Add some outliers
        n_outliers = int(n_samples * 0.1)
        outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
        X = np.vstack([X * 5, outliers])
        
    else:  # 'blobs' (default)
        # Create centers that are well separated
        centers = []
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            center_x = 4 * np.cos(angle)
            center_y = 4 * np.sin(angle)
            centers.append([center_x, center_y])
                
        X, _ = make_blobs(
            n_samples=n_samples, 
            centers=centers,
            cluster_std=noise_level*5 + 0.3,  # Scale noise level appropriately
            random_state=42
        )
    
    # Convert to list of points {x, y}
    points = [{"x": float(x), "y": float(y)} for x, y in X]
    
    return {"points": points}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(data):
    """
    Run PCA on 2D data and return the principal components, explained variance,
    and other relevant information.
    
    Parameters:
    -----------
    data : dict
        A dictionary containing:
        - X: A list of 2D points [[x1, y1], [x2, y2], ...]
        
    Returns:
    --------
    dict
        A dictionary containing:
        - components: Principal components (2x2 matrix)
        - explained_variance: Explained variance for each component
        - explained_variance_ratio: Proportion of variance explained by each component
        - mean: Mean of the data
        - original: Original data points
        - transformed: Transformed data points in PC space
        - reconstructed: Reconstructed data points using only PC1
        - corr_heatmap: Base64-encoded correlation heatmap
    """
    try:
        # Extract data points
        X = np.array(data['X'], dtype=float)
        
        # Check if we have 2D data
        if X.shape[1] != 2:
            raise ValueError("This PCA implementation only supports 2D data")
        
        # Create a DataFrame for correlation heatmap
        df = pd.DataFrame(X, columns=['X', 'Y'])
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Generate correlation heatmap
        plt.figure(figsize=(4, 3))  # Reduce the figure size
        sns.heatmap(
            corr_matrix,
            annot=True, 
            cmap='coolwarm',  # Keep the same colormap
            linewidths=0.5, 
            alpha=0.8  # Make the colors more dilute
            )
        plt.title('Correlation Matrix')
        
        # Convert heatmap to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        corr_heatmap = base64.b64encode(buf.read()).decode('utf-8')
        
        # Compute the mean of the original data
        original_mean = np.mean(X, axis=0)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create and fit PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Get principal components and variance info
        components = pca.components_
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        mean = pca.mean_

        # Reconstruct data using only PC1 (1D projection)
        X_reconstructed_1d = np.dot(X_pca[:, 0:1], components[0:1, :]) + mean

        # Transform back to original scale
        X_reconstructed_1d = scaler.inverse_transform(X_reconstructed_1d)

        # Convert to correct types for serialization
        result = {
            'components': components.tolist(),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'mean': mean.tolist(),  # Normalized mean
            'original_mean': original_mean.tolist(),  # Mean of the original data
            'original': X.tolist(),
            'transformed': X_pca.tolist(),
            'reconstructed': X_reconstructed_1d.tolist(),
            'corr_heatmap': corr_heatmap
        }

        return result
    
    except Exception as e:
        # Print the error for debugging
        import traceback
        print(f"Error in run_pca: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}

def generate_pca_data(n_samples=50, noise=0.1, seed=42):
    """
    Generate sample data for PCA visualization
    
    Parameters:
    n_samples (int): Number of samples to generate
    noise (float): Noise level to adjust the covariance matrix
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary with 'X' key containing the 2D data points
    """
    import numpy as np
    np.random.seed(seed)
    
    # Generate a correlated dataset
    mean = [0, 0]
    
    # Covariance matrix - adjust correlation based on noise
    base_correlation = 0.8
    adjusted_correlation = max(0, min(1, base_correlation - noise))
    cov = [[1.0, adjusted_correlation], [adjusted_correlation, 1.0]]
    
    # Generate random data from multivariate normal distribution
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    return {
        'X': data.tolist()
    }
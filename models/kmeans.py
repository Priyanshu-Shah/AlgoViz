import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64
from io import BytesIO
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
import io
from matplotlib.colors import ListedColormap
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

def run_kmeans(data, k=3, max_iterations=100):
    """
    Run K-means clustering algorithm
    
    Parameters:
    data (dict): Contains 'X' data points
    k (int): Number of clusters
    max_iterations (int): Maximum number of iterations
    
    Returns:
    dict: Results including centroids, labels, inertia, etc.
    """
    try:
        # Convert data to numpy arrays
        X = np.array(data['X'])
        return_history = data.get('return_history', False)
        
        # Check if data is 2D (for visualization)
        if X.shape[1] != 2:
            return {
                'error': 'K-means visualization currently only supports 2D data'
            }
        
        # Store intermediate states for animation
        animation_frames = []
        history = []
        
        # Initialize centroids randomly from the data points
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        
        # Store initial state
        labels = np.zeros(X.shape[0], dtype=int)
        animation_frames.append((centroids.copy(), labels.copy()))
        
        if return_history:
            history.append({
                'iteration': 0,
                'centroids': centroids.tolist(),
                'labels': labels.tolist()
            })
        
        # Run K-means algorithm
        converged = False
        iteration = 0
        prev_centroids = None
        
        while not converged and iteration < max_iterations:
            iteration += 1
            
            # Assign each point to the nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                     else centroids[i] for i in range(k)])
            
            # Store intermediate state
            animation_frames.append((new_centroids.copy(), labels.copy()))
            
            if return_history:
                history.append({
                    'iteration': iteration,
                    'centroids': new_centroids.tolist(),
                    'labels': labels.tolist()
                })
            
            # Check for convergence
            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, rtol=1e-4):
                converged = True
            
            prev_centroids = new_centroids
            centroids = new_centroids
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        inertia = 0
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.square(cluster_points - centroids[i]))
        
        # Calculate silhouette score if we have at least 2 clusters with data
        sil_score = None
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and all(np.sum(labels == i) > 1 for i in unique_labels):
            try:
                sil_score = silhouette_score(X, labels)
            except Exception as e:
                print(f"Error calculating silhouette score: {e}")
                sil_score = None
        
        # Prepare result
        result = {
            'centroids': centroids.tolist(),
            'labels': labels.tolist(),
            'iterations': iteration,
            'converged': converged,
            'inertia': float(inertia)
        }
        
        if sil_score is not None:
            result['silhouette_score'] = float(sil_score)
        
        if return_history:
            result['history'] = history
            
        return result
        
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_clustering_data(n_samples=100, n_clusters=3, variance=0.5, dataset_type='blobs'):
    """
    Generate sample data for clustering
    
    Parameters:
    n_samples (int): Number of samples to generate
    n_clusters (int): Number of clusters
    variance (float): Amount of noise/variance
    dataset_type (str): Type of dataset to generate (blobs, moons, circles)
    
    Returns:
    dict: Dictionary with 'X' key containing the data points
    """
    np.random.seed(42)
    
    if dataset_type == 'moons':
        # Generate two interleaving half circles
        X, _ = make_moons(n_samples=n_samples, noise=variance * 0.1)
        # Scale to -8 to 8 range
        X = X * 4 - 1.5  # Scale from default (around -1 to 1) to -8 to 8
        return {'X': X.tolist()}
    
    elif dataset_type == 'circles':
        # Generate concentric circles
        X, _ = make_circles(n_samples=n_samples, noise=variance * 0.1, factor=0.5)
        # Scale to -8 to 8 range
        X = X * 6  # Scale from default (around -1 to 1) to -8 to 8
        return {'X': X.tolist()}
    
    else:  # Default: blobs
        # Generate isotropic Gaussian blobs with centers in the range -8 to 8
        centers = np.random.uniform(-7, 7, (n_clusters, 2))  # Generate random centers in the -6 to 6 range
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=variance * 2,  # Increase standard deviation to match the larger scale
            random_state=42
        )
        
    
            
        return {'X': X.tolist()}
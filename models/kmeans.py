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
    
    try:
        # Convert data to numpy arrays
        X = np.array(data['X'])
        
        # Check if data is 2D (for visualization)
        if X.shape[1] != 2:
            return {
                'error': 'K-means visualization currently only supports 2D data'
            }
        
        # Store intermediate states for animation
        animation_frames = []
        
        # Initialize centroids randomly from the data points
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        
        # Store initial state
        labels = np.zeros(X.shape[0], dtype=int)
        animation_frames.append((centroids.copy(), labels.copy()))
        
        # Run K-means algorithm
        converged = False
        iteration = 0
        prev_centroids = None
        
        while not converged and iteration < max_iterations:
            # Assign each point to the nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                     else centroids[i] for i in range(k)])
            
            # Store intermediate state
            animation_frames.append((new_centroids.copy(), labels.copy()))
            
            # Check for convergence
            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, rtol=1e-4):
                converged = True
            
            prev_centroids = new_centroids
            centroids = new_centroids
            iteration += 1
        
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
        
        # Create final plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        
        # Plot points with cluster colors
        for i in range(k):
            ax.scatter(X[labels == i, 0], X[labels == i, 1], s=50, color=colors[i], alpha=0.7, label=f'Cluster {i+1}')
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', color='red', label='Centroids')
        
        ax.set_title(f'K-Means Clustering (k={k}, iterations={iteration})')
        ax.set_xlabel('Feature 1', fontsize=12, labelpad=10)
        ax.set_ylabel('Feature 2', fontsize=12, labelpad=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Add padding to the plot
        x_range = X[:, 0].max() - X[:, 0].min()
        y_range = X[:, 1].max() - X[:, 1].min()
        
        # 20% padding on each side
        ax.set_xlim(
            X[:, 0].min() - 0.2 * x_range,
            X[:, 0].max() + 0.2 * x_range
        )
        ax.set_ylim(
            X[:, 1].min() - 0.2 * y_range,
            X[:, 1].max() + 0.2 * y_range
        )
        
        plt.tight_layout(pad=2.0)  # Add extra padding around the entire plot
        
        # Convert plot to base64 string
        plot_bytes = BytesIO()
        plt.tight_layout()
        plt.savefig(plot_bytes, format='png', dpi=100)
        plt.close(fig)
        plot_bytes.seek(0)
        plot_base64 = base64.b64encode(plot_bytes.read()).decode('utf-8')
        
        # Create animation
        animation_base64 = None
        try:
            if len(animation_frames) > 2:
                animation_base64 = create_animation(X, animation_frames)
        except Exception as e:
            print(f"Error creating animation: {e}")
        
        # Prepare result
        result = {
            'centroids': centroids.tolist(),
            'labels': labels.tolist(),
            'iterations': iteration,
            'converged': converged,
            'inertia': float(inertia),
            'plot': plot_base64
        }
        
        if sil_score is not None:
            result['silhouette_score'] = float(sil_score)
            
        if animation_base64 is not None:
            result['animation'] = animation_base64
            
        return result
        
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def create_animation(X, frames):
    """Create animation of K-means process"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame_idx):
        ax.clear()
        centroids, labels = frames[frame_idx]
        k = centroids.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        
        # Plot points with cluster colors
        for i in range(k):
            ax.scatter(X[labels == i, 0], X[labels == i, 1], s=50, color=colors[i], alpha=0.7)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', color='red')
        
        ax.set_title(f'K-Means Clustering - Iteration {frame_idx}')
        ax.set_xlabel('Feature 1', fontsize=12, labelpad=8)
        ax.set_ylabel('Feature 2', fontsize=12, labelpad=8)
        ax.grid(alpha=0.3)
        
        # Set consistent axis limits with padding
        all_x = X[:, 0]
        all_y = X[:, 1]
        margin_x = (max(all_x) - min(all_x)) * 0.2
        margin_y = (max(all_y) - min(all_y)) * 0.2
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    # Create animation (max 20 frames to keep file size reasonable)
    frames_to_use = frames
    if len(frames) > 20:
        # Use evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, 20).astype(int)
        frames_to_use = [frames[i] for i in indices]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames_to_use), interval=500)
    
    # Save animation to bytes
    animation_bytes = BytesIO()
    ani.save(animation_bytes, writer='pillow', fps=2, dpi=80)
    plt.close(fig)
    
    # Convert to base64
    animation_bytes.seek(0)
    animation_base64 = base64.b64encode(animation_bytes.read()).decode('utf-8')
    
    return animation_base64

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
        # Return only the features since we don't need the true labels for clustering
        return {'X': X.tolist()}
    
    elif dataset_type == 'circles':
        # Generate concentric circles
        X, _ = make_circles(n_samples=n_samples, noise=variance * 0.1, factor=0.5)
        return {'X': X.tolist()}
    
    else:  # Default: blobs
        # Generate isotropic Gaussian blobs
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            cluster_std=variance,
            random_state=42
        )
        return {'X': X.tolist()}

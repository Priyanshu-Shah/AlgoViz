import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from sklearn.cluster import DBSCAN
import base64
from io import BytesIO
import time

import numpy as np

def run_dbscan(points, eps=0.5, min_samples=5, 
               show_core_points=True, show_border_points=True, 
               show_noise_points=True, show_epsilon_radius=True):
    """
    Run the DBSCAN algorithm and return the results along with step-by-step iterations.
    
    Args:
        points: List of dicts with x and y coordinates
        eps: Epsilon parameter (radius for neighbor search)
        min_samples: Minimum number of points to form a dense region
        show_core_points: Whether to highlight core points (param passed to frontend)
        show_border_points: Whether to highlight border points (param passed to frontend)
        show_noise_points: Whether to show noise points (param passed to frontend)
        show_epsilon_radius: Whether to show epsilon radius (param passed to frontend)
    
    Returns:
        Dictionary with iterations and clustering results
    """
    try:
        # Convert input points to numpy array
        X = np.array([[point['x'], point['y']] for point in points])
        
        # Create an interactive DBSCAN object
        interactive_dbscan = InteractiveDBSCAN(eps=eps, min_samples=min_samples)
        
        # Run the algorithm and collect iterations
        iterations = interactive_dbscan.run(X)
        
        # Return the results with all NumPy types converted to Python native types
        result = {
            "status": "success",
            "iterations": iterations,
            "final_labels": interactive_dbscan.labels_.tolist() if hasattr(interactive_dbscan, 'labels_') else [],
            "num_clusters": len(set(interactive_dbscan.labels_)) - (1 if -1 in interactive_dbscan.labels_ else 0) if hasattr(interactive_dbscan, 'labels_') else 0,
            "core_points": [int(x) for x in interactive_dbscan.core_points],
            "border_points": [int(x) for x in interactive_dbscan.border_points],
            "noise_points": [int(x) for x in interactive_dbscan.noise_points]
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in run_dbscan: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return {"status": "error", "message": str(e), "traceback": error_traceback}

class InteractiveDBSCAN:
    """
    A custom implementation of DBSCAN algorithm that captures intermediate steps
    for visualization.
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize the DBSCAN algorithm with parameters.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.X = None
        self.labels_ = None
        self.core_points = []
        self.border_points = []
        self.noise_points = []
        
    def run(self, X):
        """
        Run the DBSCAN algorithm and collect intermediate steps.
        
        Args:
            X: Input data as a numpy array of shape (n_samples, n_features)
            
        Returns:
            List of iterations with visualization and metadata
        """
        self.X = X
        n_samples = X.shape[0]
        
        # Initialize all points as unvisited (-2 means unvisited)
        self.labels_ = np.full(n_samples, -2, dtype=int)
        
        # Keep track of iterations
        iterations = []
        
        # Add initial state
        iterations.append(self._create_iteration_data(
            phase="initialization",
            labels=self.labels_.tolist(),
            current_point=None,
            neighbors=[],
            message="Algorithm initialized. Ready to start clustering."
        ))
        
        # Initialize cluster label
        cluster_id = 0
        
        # Process each point
        for point_idx in range(n_samples):
            # Skip already processed points
            if self.labels_[point_idx] != -2:
                continue
            
            # Find neighbors
            neighbors = self._find_neighbors(point_idx)
            
            # Add iteration showing current point evaluation
            iterations.append(self._create_iteration_data(
                phase="evaluating_point",
                labels=self.labels_.tolist(),
                current_point=point_idx,
                neighbors=neighbors.tolist(),
                message=f"Evaluating point {point_idx}. Found {len(neighbors)} neighbors."
            ))
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # -1 means noise
                self.noise_points.append(point_idx)
                
                # Add iteration showing noise point identification
                iterations.append(self._create_iteration_data(
                    phase="noise_identified",
                    labels=self.labels_.tolist(),
                    current_point=point_idx,
                    neighbors=neighbors.tolist(),
                    message=f"Point {point_idx} classified as noise (insufficient neighbors)."
                ))
                continue
            
            # Start a new cluster
            cluster_id += 1
            self.labels_[point_idx] = cluster_id - 1  # Cluster labels start from 0
            self.core_points.append(point_idx)
            
            # Add iteration showing new cluster formation
            iterations.append(self._create_iteration_data(
                phase="new_cluster_formed",
                labels=self.labels_.tolist(),
                current_point=point_idx,
                neighbors=neighbors.tolist(),
                message=f"New cluster {cluster_id-1} started with core point {point_idx}."
            ))
            
            # Process neighbors
            seed_points = list(neighbors)
            processed_indices = []
            
            # Expand cluster
            while seed_points:
                current_seed = seed_points.pop(0)
                
                # Skip already processed points
                if current_seed in processed_indices:
                    continue
                processed_indices.append(current_seed)
                
                # Skip points already assigned to clusters or marked as noise
                if self.labels_[current_seed] != -2:
                    continue
                
                # Mark neighbor as part of cluster
                self.labels_[current_seed] = cluster_id - 1
                
                # Find neighbor's neighbors
                neighbor_neighbors = self._find_neighbors(current_seed)
                
                # Add iteration showing neighbor evaluation
                iterations.append(self._create_iteration_data(
                    phase="evaluating_neighbor",
                    labels=self.labels_.tolist(),
                    current_point=current_seed,
                    neighbors=neighbor_neighbors.tolist(),
                    message=f"Evaluating neighbor point {current_seed}. Found {len(neighbor_neighbors)} neighbors."
                ))
                
                # If neighbor has enough neighbors, add its neighbors to seed points and mark as core point
                if len(neighbor_neighbors) >= self.min_samples:
                    self.core_points.append(current_seed)
                    
                    # Add new neighbors to seeds for further expansion
                    for secondary_neighbor in neighbor_neighbors:
                        if self.labels_[secondary_neighbor] == -2 and secondary_neighbor not in processed_indices:
                            seed_points.append(secondary_neighbor)
                            
                    # Add iteration showing core point identification
                    iterations.append(self._create_iteration_data(
                        phase="expanding_cluster",
                        labels=self.labels_.tolist(),
                        current_point=current_seed,
                        neighbors=neighbor_neighbors.tolist(),
                        message=f"Expanding cluster {cluster_id-1} with core point {current_seed}."
                    ))
                else:
                    # Not enough neighbors, so this is a border point
                    self.border_points.append(current_seed)
                    
                    # Add iteration showing border point identification
                    iterations.append(self._create_iteration_data(
                        phase="border_point_identified",
                        labels=self.labels_.tolist(),
                        current_point=current_seed,
                        neighbors=neighbor_neighbors.tolist(),
                        message=f"Point {current_seed} classified as border point in cluster {cluster_id-1}."
                    ))
        
        # Post-process to ensure we don't have duplicate core/border/noise points
        self.core_points = list(set(self.core_points))
        self.border_points = list(set(self.border_points))
        self.noise_points = list(set(self.noise_points))
        
        # Add final state
        iterations.append(self._create_iteration_data(
            phase="completed",
            labels=self.labels_.tolist(),
            current_point=None,
            neighbors=[],
            message=f"Clustering complete! Found {cluster_id} clusters, {len(self.core_points)} core points, {len(self.border_points)} border points, and {len(self.noise_points)} noise points."
        ))
        
        return iterations
    
    def _find_neighbors(self, point_idx):
        """Find all points within eps distance of a point."""
        distances = np.sqrt(np.sum((self.X - self.X[point_idx])**2, axis=1))
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors
    
    def _create_iteration_data(self, phase, labels, current_point, neighbors, message=""):
        """Create a data structure for an iteration."""
        # Convert NumPy types to Python native types
        processed_labels = [int(l) for l in labels]
        processed_neighbors = [int(n) for n in neighbors] if neighbors is not None else []
        
        # Calculate processed points count
        processed_count = sum(1 for l in labels if l != -2)
        
        # Count clusters
        cluster_labels = set(l for l in labels if l >= 0)
        cluster_count = len(cluster_labels)
        
        # Return iteration data
        return {
            "phase": phase,
            "labels": processed_labels,
            "current_point": int(current_point) if current_point is not None else None,
            "neighbors": processed_neighbors,
            "core_points": [int(cp) for cp in self.core_points],
            "border_points": [int(bp) for bp in self.border_points],
            "noise_points": [int(np) for np in self.noise_points],
            "num_clusters": int(cluster_count),
            "num_processed": int(processed_count),
            "message": message
        }

import numpy as np

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
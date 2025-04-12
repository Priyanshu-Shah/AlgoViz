import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    """
    Neural network implementation from scratch using NumPy
    """
    def __init__(self, input_size, hidden_layers, output_size, output_activation='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = output_activation
        
        # Initialize architecture
        self.layer_sizes = [input_size] + [layer['neurons'] for layer in hidden_layers] + [output_size]
        self.activations = [layer['activation'] for layer in hidden_layers] + [output_activation]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Xavier/Glorot initialization for weights
        for i in range(len(self.layer_sizes) - 1):
            # Calculate weight scale based on input and output size of the layer
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            
            # Initialize weights with random values scaled appropriately
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale)
            
            # Initialize biases with zeros
            self.biases.append(np.zeros(self.layer_sizes[i+1]))
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Parameters:
        X: Input data, shape (batch_size, input_size)
        
        Returns:
        tuple: (output, cache) where cache contains intermediate values for backprop
        """
        # Initialize cache for storing intermediate values
        cache = {'A': [X], 'Z': []}
        
        # Forward through each layer
        A = X
        for i in range(len(self.weights)):
            # Linear transformation Z = A * W + b
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            cache['Z'].append(Z)
            
            # Apply activation function
            A = self._activate(Z, self.activations[i])
            cache['A'].append(A)
        
        # Return output and cache
        return A, cache
    
    def backward(self, X, y, cache, learning_rate):
        """
        Backward pass through the network (backpropagation)
        
        Parameters:
        X: Input data
        y: Target labels
        cache: Cached values from forward pass
        learning_rate: Learning rate for gradient descent
        
        Returns:
        float: Loss value
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Calculate output layer error based on activation function
        output = cache['A'][-1]
        
        if self.output_activation == 'sigmoid':
            # Binary cross entropy loss and gradient for sigmoid
            loss = -np.mean(y * np.log(output + 1e-10) + (1 - y) * np.log(1 - output + 1e-10))
            dZ = output - y
        else:  # softmax
            # Cross entropy loss and gradient for softmax
            one_hot_y = self._one_hot_encode(y, self.output_size)
            loss = -np.mean(np.sum(one_hot_y * np.log(output + 1e-10), axis=1))
            dZ = output - one_hot_y
        
        # Backpropagate through the layers
        for l in reversed(range(len(self.weights))):
            # Calculate gradients for this layer
            dW[l] = np.dot(cache['A'][l].T, dZ) / m
            db[l] = np.sum(dZ, axis=0) / m
            
            # Calculate error for previous layer (if not at input layer)
            if l > 0:
                dA = np.dot(dZ, self.weights[l].T)
                dZ = dA * self._activate_backward(cache['Z'][l-1], self.activations[l-1])
        
        # Update weights and biases
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * dW[l]
            self.biases[l] -= learning_rate * db[l]
        
        return loss
    
    def _activate(self, Z, activation):
        """
        Apply activation function
        
        Parameters:
        Z: Input to activation function
        activation: Name of activation function
        
        Returns:
        numpy.ndarray: Output after applying activation function
        """
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to prevent overflow
        elif activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            # Subtract max for numerical stability
            Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        return Z
    
    def _activate_backward(self, Z, activation):
        """
        Calculate gradient of activation function
        
        Parameters:
        Z: Input to activation function
        activation: Name of activation function
        
        Returns:
        numpy.ndarray: Gradient of activation function
        """
        if activation == 'sigmoid':
            A = self._activate(Z, 'sigmoid')
            return A * (1 - A)
        elif activation == 'relu':
            return (Z > 0).astype(float)
        elif activation == 'tanh':
            A = np.tanh(Z)
            return 1 - np.square(A)
        # For softmax, we handle this directly in backward()
        return np.ones_like(Z)
    
    def _one_hot_encode(self, y, num_classes):
        """
        Convert class labels to one-hot encoded vectors
        
        Parameters:
        y: Class labels
        num_classes: Number of classes
        
        Returns:
        numpy.ndarray: One-hot encoded vectors
        """
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot
    
    def train(self, X, y, learning_rate, epochs, batch_size=32, validation_data=None, verbose=0):
        """
        Train the neural network
        
        Parameters:
        X: Training data
        y: Training labels
        learning_rate: Learning rate for gradient descent
        epochs: Number of epochs to train
        batch_size: Mini-batch size
        validation_data: Tuple of (X_val, y_val) for validation
        verbose: Verbosity level (0: silent, 1: report every 10 epochs)
        
        Returns:
        dict: Training history
        """
        m = X.shape[0]
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print("Training epochs:", epochs)

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            epoch_loss = 0
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output, cache = self.forward(X_batch)
                
                # Backward pass
                batch_loss = self.backward(X_batch, y_batch, cache, learning_rate)
                epoch_loss += batch_loss * X_batch.shape[0]
            
            # Calculate average loss and accuracy for the epoch
            epoch_loss /= m
            train_accuracy = self.evaluate(X, y)
            history['loss'].append(epoch_loss)
            history['accuracy'].append(train_accuracy)
            
            # Calculate validation metrics if validation data is provided
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output, _ = self.forward(X_val)
                
                if self.output_activation == 'sigmoid':
                    val_loss = -np.mean(y_val * np.log(val_output + 1e-10) + 
                                     (1 - y_val) * np.log(1 - val_output + 1e-10))
                else:
                    one_hot_y_val = self._one_hot_encode(y_val, self.output_size)
                    val_loss = -np.mean(np.sum(one_hot_y_val * np.log(val_output + 1e-10), axis=1))
                
                val_accuracy = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
                val_info = f", val_loss: {history['val_loss'][-1]:.4f}, val_acc: {history['val_accuracy'][-1]:.4f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss:.4f}, accuracy: {train_accuracy:.4f}{val_info}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        X: Input data
        
        Returns:
        numpy.ndarray: Predicted class labels
        """
        # Forward pass
        output, _ = self.forward(X)
        
        # Convert outputs to class predictions
        if self.output_activation == 'sigmoid':
            return (output > 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class
        
        Parameters:
        X: Input data
        
        Returns:
        numpy.ndarray: Probability estimates
        """
        # Forward pass
        output, _ = self.forward(X)
        return output
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy
        
        Parameters:
        X: Input data
        y: True labels
        
        Returns:
        float: Accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.flatten())

def run_ann(data):
    """
    Train a neural network on the input data and return training results.

    Parameters:
    data (dict): Input data containing:
        - X: List of data points (each point is a dict with 'x' and 'y', or a list [x, y]).
        - y: List of class labels.
        - architecture: Dict with network architecture details:
            - hiddenLayers: List of dicts with 'neurons' and 'activation' for each layer
            - learningRate: Learning rate for the optimizer
            - epochs: Number of training epochs
            - outputActivation: Activation function for the output layer ('sigmoid' or 'softmax')

    Returns:
    dict: Results including:
        - accuracy: Training accuracy of the model
        - loss: Final loss value
        - decision_boundary: Base64-encoded image of the decision boundary visualization
        - model_info: Dict with training details
        - epochs: Number of epochs trained
    """
    try:
        # Extract data
        X_raw = data.get('X', [])
        y_raw = data.get('y', [])
        architecture = data.get('network_config', {})
        
        # Handle different input formats
        X = []
        for point in X_raw:
            if isinstance(point, dict):
                # If point is a dictionary with 'x' and 'y' keys
                if 'x' in point and 'y' in point:
                    X.append([float(point['x']), float(point['y'])])
                elif 'x1' in point and 'x2' in point:
                    X.append([float(point['x1']), float(point['x2'])])
            elif isinstance(point, list):
                # If point is already a list [x, y]
                X.append([float(point[0]), float(point[1])])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y_raw)
        
        # Normalize input data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get architecture parameters
        hidden_layers = architecture.get('hiddenLayers', [{'neurons': 10, 'activation': 'relu'}])
        learning_rate = float(architecture.get('learningRate', 0.01))
        epochs = int(architecture.get('epochs', 100))
        output_activation = architecture.get('outputActivation', 'softmax')  # Default to softmax for multi-class
        
        # Get the number of unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Always use multiple output units for softmax
        output_units = n_classes
        
        # Create and train the model
        model = NeuralNetwork(
            input_size=2,
            hidden_layers=hidden_layers,
            output_size=output_units,
            output_activation=output_activation
        )
        
        # Create a small validation split
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Prepare target format based on activation
        # If using softmax (multi-class), no need to reshape y
        # If using sigmoid (binary), reshape y to have shape (n_samples, 1)
        if output_activation == 'sigmoid' and output_units == 1:
            y_train_formatted = y_train.reshape(-1, 1)
            y_val_formatted = y_val.reshape(-1, 1)
        else:
            y_train_formatted = y_train
            y_val_formatted = y_val
        
        # Train the model
        history = model.train(
            X=X_train, 
            y=y_train_formatted,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_val, y_val_formatted),
            verbose=0
        )
        
        # Calculate final accuracy
        accuracy = model.evaluate(X_scaled, y_train_formatted if output_activation == 'sigmoid' and output_units == 1 else y)
        
        # Generate decision boundary visualization
        decision_boundary = generate_decision_boundary(model, X, y, scaler)

        # Generate neuron feature visualizations
        neuron_visualizations = generate_neuron_visualizations(model, scaler)
        
        # Collect training history
        train_loss = history['loss'][-1] if len(history['loss']) > 0 else 0
        train_accuracy = history['accuracy'][-1] if len(history['accuracy']) > 0 else 0
        val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else 0
        val_accuracy = history['val_accuracy'][-1] if len(history['val_accuracy']) > 0 else 0

        training_history = []
        for i in range(len(history['loss'])):
            epoch_data = {
                'epoch': i+1,
                'loss': float(history['loss'][i]),
                'accuracy': float(history['accuracy'][i])
            }
            if len(history['val_loss']) > i:
                epoch_data['val_loss'] = float(history['val_loss'][i])
                epoch_data['val_accuracy'] = float(history['val_accuracy'][i])
            training_history.append(epoch_data)
        
        return {
            'accuracy': float(accuracy),
            'loss': float(train_loss),
            'decision_boundary': decision_boundary,
            'epochs': epochs,
            'model_info': {
                'architecture': [l.get('neurons', 10) for l in hidden_layers],
                'activations': [l.get('activation', 'relu') for l in hidden_layers],
                'output_activation': output_activation,
                'learning_rate': learning_rate,
                'train_accuracy': float(train_accuracy),
                'train_loss': float(train_loss),
                'val_accuracy': float(val_accuracy),
                'val_loss': float(val_loss),
                'classes': unique_classes.tolist(),
                'training_history': training_history
            },
            'weights': [w.tolist() for w in model.weights],
            'biases': [b.tolist() for b in model.biases],
            'neuron_visualizations': neuron_visualizations
        }
    
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_decision_boundary(model, X, y, scaler=None):
    """
    Generate a visualization of the decision boundary for the trained neural network.
    
    Parameters:
    model: Trained neural network model
    X: Training data points
    y: Training data labels
    scaler: Optional standardization scaler used during training
    
    Returns:
    str: Base64 encoded image of the decision boundary
    """
    try:
        # Determine the plotting bounds
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Create a mesh grid
        h = 0.05  # Step size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Flatten the mesh grid and reshape for prediction
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Apply the same scaling as during training if provided
        if scaler is not None:
            mesh_points_scaled = scaler.transform(mesh_points)
        else:
            mesh_points_scaled = mesh_points
        
        # For multi-class, we need the raw probabilities, not just the class prediction
        if model.output_size > 1:
            # Get raw probabilities
            Z_proba, _ = model.forward(mesh_points_scaled)
            # Get the predicted class (argmax)
            Z = np.argmax(Z_proba, axis=1)
        else:
            # For binary classification
            Z = model.predict(mesh_points_scaled)
        
        # Reshape to match the mesh grid
        Z = Z.reshape(xx.shape)
        
        # Create the plot
        plt.figure(figsize=(8, 6), dpi=100)
        
        # Get unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Define colors for different classes based on number of classes
        if n_classes <= 2:
            # For binary classification
            cmap_light = plt.cm.get_cmap('coolwarm', 2)
            colors = plt.cm.coolwarm(np.linspace(0, 1, n_classes))
        else:
            # For multi-class classification
            cmap_light = plt.cm.get_cmap('viridis', n_classes)
            colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        
        # Plot the decision boundary
        if n_classes > 2:
            # Create levels for each class boundary
            levels = np.arange(n_classes + 1) - 0.5
            plt.contourf(xx, yy, Z, levels=levels, alpha=0.3, cmap=cmap_light)
        else:
            # For binary classification
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        
        # Plot the training points with different markers for each class
        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
        
        for i, cls in enumerate(unique_classes):
            idx = np.where(y == cls)
            plt.scatter(X[idx, 0], X[idx, 1], 
                       c=[colors[i]], 
                       marker=markers[i % len(markers)],
                       edgecolor='black',
                       s=50,
                       label=f'Class {cls}')
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Neural Network Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()
        
        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        
        # Encode the image
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
    
    except Exception as e:
        import traceback
        print(f"Error generating decision boundary: {str(e)}")
        print(traceback.format_exc())
        
        # Return an error image
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error generating decision boundary: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64

def predict_points(model_results, points, scaler=None):
    """
    Predict classes for new points using the model information from training.
    
    Parameters:
    model_results: Dictionary containing model information from training
    points: List of points to predict
    scaler: Optional standardization scaler used during training
    
    Returns:
    list: Predicted classes for each point
    """
    try:
        # Convert points to numpy array
        X = np.array(points)
        
        # Apply scaling if provided
        if scaler is not None:
            X = scaler.transform(X)
            
        # Extract model architecture from the results
        architecture = model_results.get('model_info', {})
        hidden_layers = []
        
        # Reconstruct the layers from the model info
        for i, neurons in enumerate(architecture.get('architecture', [])):
            hidden_layers.append({
                'neurons': neurons,
                'activation': architecture.get('activations', [])[i] if i < len(architecture.get('activations', [])) else 'relu'
            })
            
        output_activation = architecture.get('output_activation', 'softmax')
        
        # Get number of classes from the model info or default to 3
        classes = architecture.get('classes', [0, 1, 2])
        n_classes = len(classes)
        
        # Create a new model with the same architecture
        model = NeuralNetwork(
            input_size=2,
            hidden_layers=hidden_layers,
            output_size=n_classes,
            output_activation=output_activation
        )
        
        # Set the weights and biases from the saved model
        if 'weights' in model_results and 'biases' in model_results:
            # Convert lists back to numpy arrays
            model.weights = [np.array(w) for w in model_results['weights']]
            model.biases = [np.array(b) for b in model_results['biases']]
            
            # Make predictions
            predictions = model.predict(X)
            return predictions.tolist()
        else:
            # If no weights/biases provided, return random predictions as a fallback
            print("Warning: No model weights/biases found. Using random predictions.")
            return np.random.randint(0, n_classes, size=len(points)).tolist()
        
    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        # Return default predictions (all zeros) in case of error
        return [0] * len(points)

def generate_sample_data(dataset_type='blobs', n_samples=100, n_clusters=2, variance=0.5):
    """
    Generate sample data for neural network training.
    
    Parameters:
    dataset_type (str): Type of dataset ('blobs', 'moons', 'circles', 'xor', 'spiral')
    n_samples (int): Number of samples to generate
    n_clusters (int): Number of clusters/classes
    variance (float): Noise level
    
    Returns:
    dict: Dictionary containing X features and y labels
    """
    from sklearn.datasets import make_blobs, make_moons, make_circles
    
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=variance*0.1, random_state=42)
        # Scale to reasonable range
        X = X * 6 - 3
    
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=variance*0.1, factor=0.5, random_state=42)
        # Scale to reasonable range
        X = X * 6
    
    elif dataset_type == 'xor':
        # Generate XOR pattern
        n_per_cluster = n_samples // 4  # 4 clusters for XOR
        
        # Generate points in the four quadrants
        X1 = np.random.randn(n_per_cluster, 2) * variance + np.array([3, 3])  # top-right
        X2 = np.random.randn(n_per_cluster, 2) * variance + np.array([-3, 3])  # top-left
        X3 = np.random.randn(n_per_cluster, 2) * variance + np.array([3, -3])  # bottom-right
        X4 = np.random.randn(n_per_cluster, 2) * variance + np.array([-3, -3])  # bottom-left
        
        # Combine all points
        X = np.vstack([X1, X2, X3, X4])
        
        # Assign labels (XOR pattern: same label for opposite quadrants)
        y = np.array([0] * n_per_cluster + [1] * n_per_cluster + [0] * n_per_cluster + [1] * n_per_cluster)
        
        # Shuffle the data
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
    
    elif dataset_type == 'spiral':
        # Generate spiral pattern
        n_per_class = n_samples // n_clusters
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_clusters):
            ix = range(n_per_class * i, n_per_class * (i + 1))
            r = np.linspace(0.0, 8.0, n_per_class)  # radius
            t = np.linspace(i * 4, (i + 1) * 4, n_per_class) + np.random.randn(n_per_class) * variance  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = i
            
    else:  # 'blobs' (default)
        centers = []
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            r = 4  # radius
            center_x = r * np.cos(angle)
            center_y = r * np.sin(angle)
            centers.append([center_x, center_y])
        
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=variance*1.5,
            random_state=42
        )
    
    # Convert to lists for JSON serialization
    return {
        'X': X.tolist(),
        'y': y.tolist()
    }

def generate_neuron_visualizations(model, scaler=None):
    """
    Generate visualizations of what each neuron in the network has learned.
    
    Parameters:
    model: Trained neural network model
    scaler: Optional standardization scaler used during training
    
    Returns:
    dict: Dictionary of base64 encoded images for each layer's neurons
    """
    try:
        # Create a mesh grid covering our input space
        x_min, x_max = -8, 8
        y_min, y_max = -8, 8
        h = 0.1  # Step size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Flatten the mesh grid and reshape for forward pass
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Apply the same scaling as during training if provided
        if scaler is not None:
            mesh_points_scaled = scaler.transform(mesh_points)
        else:
            mesh_points_scaled = mesh_points
        
        # Dictionary to store base64 visualizations
        visualizations = {}
        
        # Print debug info
        print(f"Model weights shape: {[w.shape for w in model.weights]}")
        print(f"Model has {len(model.weights)-1} hidden layers")
        
        # For each layer, we'll generate activations and visualize them
        activations = [mesh_points_scaled]  # Start with input
        
        # IMPORTANT: First compute all activations through the entire network
        # This ensures we have proper propagation through all layers
        for layer_idx in range(len(model.weights)):
            W = model.weights[layer_idx]
            b = model.biases[layer_idx]
            
            # Linear transformation
            Z = np.dot(activations[-1], W) + b
            
            # Apply activation
            activation_fn = model.activations[layer_idx]
            A = model._activate(Z, activation_fn)
            activations.append(A)
            
            print(f"Computing activations for layer {layer_idx}: {A.shape}")
            
        # Now generate visualizations for each hidden layer
        # We exclude the output layer (last one)
        for layer_idx in range(len(model.weights) - 1):  # Exclude output layer
            print(f"Generating visualizations for layer {layer_idx}")
            
            # Get the activations for this layer
            A = activations[layer_idx + 1]  # +1 because activations[0] is the input
            
            # Generate visualization for each neuron in this layer
            neuron_images = []
            for neuron_idx in range(A.shape[1]):
                # Extract activation values for this neuron
                neuron_activation = A[:, neuron_idx].reshape(xx.shape)
                
                # Create normalized visualization
                plt.figure(figsize=(3, 3), dpi=100)
                plt.axis('off')
                
                # Use a perceptually uniform colormap
                vmin = np.percentile(neuron_activation, 5)
                vmax = np.percentile(neuron_activation, 95)
                plt.imshow(neuron_activation, cmap='viridis', origin='lower', 
                          extent=[x_min, x_max, y_min, y_max], vmin=vmin, vmax=vmax)
                
                # Convert to base64 image
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                neuron_images.append(image_base64)
            
            # Store this layer's neuron visualizations
            visualizations[f'layer_{layer_idx}'] = neuron_images
            print(f"Added {len(neuron_images)} visualizations for layer_{layer_idx}")
        
        # Final debug info 
        print(f"Final visualizations: {list(visualizations.keys())}")
        for layer_key, images in visualizations.items():
            print(f"Layer {layer_key}: {len(images)} neuron visualizations")
        
        return visualizations
    
    except Exception as e:
        import traceback
        print(f"Error generating neuron visualizations: {str(e)}")
        print(traceback.format_exc())
        return {}
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def run_svm(data):
    """
    Perform SVM on the input data and return results for decision boundary and support vectors.

    Parameters:
    data (dict): Input data containing:
        - X: List of data points (each point is a dict with 'x' and 'y').
        - y: List of class labels.
        - kernel: Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid').

    Returns:
    dict: Results containing:
        - decision_boundary: Base64-encoded image of the decision boundary.
        - support_vectors: List of support vectors.
        - accuracy: Training accuracy of the SVM model.
    """
    try:
        # Extract data
        X = np.array([[point['x'], point['y']] for point in data['X']])
        y = np.array(data['y'])
        kernel = data.get('kernel', 'linear')

        # Create and train the SVM model
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=True))
        model.fit(X, y)

        # Get support vectors
        support_vectors = model.named_steps['svc'].support_vectors_.tolist()

        # Calculate accuracy
        accuracy = model.score(X, y)

        # Generate decision boundary
        decision_boundary = generate_decision_boundary(model, X, y)

        return {
            'decision_boundary': decision_boundary,
            'support_vectors': support_vectors,
            'accuracy': accuracy
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def generate_decision_boundary(model, X, y):
    """
    Generate a decision boundary plot for the SVM model.

    Parameters:
    model: Trained SVM model.
    X: Training data points.
    y: Class labels.

    Returns:
    str: Base64-encoded image of the decision boundary.
    """
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.scatter(model.named_steps['svc'].support_vectors_[:, 0],
                model.named_steps['svc'].support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')
    plt.title('SVM Decision Boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
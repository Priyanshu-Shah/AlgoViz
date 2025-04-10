import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

function Docs() {
  return (
    <motion.div 
      className="model-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <style>
        {`
          .docs-container {
            font-family: 'Arial', sans-serif;
            color: #333;
            line-height: 1.8;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
          }

          .section-title {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-bottom: 1rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
          }

          .model-title {
            font-size: 2rem;
            color: #2980b9;
            text-align: center;
            margin-bottom: 2rem;
          }

          .content-container {
            display: flex;
            flex-direction: column;
            gap: 2rem;
          }

          .input-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
          }

          h3 {
            font-size: 1.2rem;
            color: #34495e;
          }

          p {
            font-size: 1rem;
            color: #7f8c8d;
          }

          ol {
            padding-left: 1.5rem;
            list-style-type: decimal;
          }

          ol li {
            margin-bottom: 0.5rem;
          }
        `}
      </style>

      <div className="docs-container">
        <h1 className="model-title">Documentation</h1>

        <div className="content-container">
          <div className="input-section">
            <h2 className="section-title">Getting Started</h2>
            <p>
              This documentation will help you understand how to use the ML Visualizer
              application effectively. Choose a model from the home page or navigation
              bar to get started.
            </p>
          </div>

          <div className="input-section">
            <h2 className="section-title">Available Models</h2>
            <div>
              <h3>Linear Regression</h3>
              <p>
                Input pairs of X and Y values to visualize a linear relationship between
                variables. The model will calculate the best-fitting line and various
                metrics.
              </p>

              <h3>K-Nearest Neighbors (KNN)</h3>
              <p>
                Choose between classification or regression. For classification, input
                points with class labels. For regression, input X-Y pairs. The model
                will make predictions based on the nearest neighbors.
              </p>

              <h3>Decision Trees</h3>
              <p>
                Decision Trees split the data into subsets based on feature values,
                creating a tree structure. Each node represents a decision rule, and
                leaves represent outcomes.
              </p>

              <h3>K-Means Clustering</h3>
              <p>
                K-Means is an unsupervised algorithm that partitions data into K
                clusters. It iteratively assigns points to the nearest cluster centroid
                and updates centroids until convergence.
              </p>

              <h3>DBSCAN</h3>
              <p>
                DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
                groups points that are closely packed together while marking outliers as
                noise. It uses density-based criteria for clustering.
              </p>

              <h3>Principal Component Analysis (PCA)</h3>
              <p>
                PCA reduces the dimensionality of data by transforming it into a new set
                of orthogonal components that capture the maximum variance in the data.
              </p>

              <h3>Artificial Neural Networks (ANN)</h3>
              <p>
                ANNs are computational models inspired by the human brain. They consist
                of layers of interconnected nodes (neurons) that process data and learn
                patterns through backpropagation.
              </p>

              <h3>Support Vector Machines (SVM)</h3>
              <p>
                SVMs are supervised learning models that find the optimal hyperplane to
                separate data into classes. They can use different kernels to handle
                linear and non-linear data.
              </p>
            </div>
          </div>

          <div className="input-section">
            <h2 className="section-title">How to Use</h2>
            <ol>
              <li>Select a machine learning model from the home page</li>
              <li>Input your data (or load sample data when available)</li>
              <li>Configure any model-specific parameters</li>
              <li>Run the model to see results and visualizations</li>
              <li>Analyze the output metrics and visualization</li>
            </ol>
          </div>

          <div className="input-section">
            <h2 className="section-title">How It Works</h2>
            <div>
              <h3>Linear Regression</h3>
              <p>
                Linear Regression models the relationship between a dependent variable
                and one or more independent variables using a linear equation. It
                minimizes the sum of squared errors to find the best-fitting line.
              </p>

              <h3>K-Nearest Neighbors (KNN)</h3>
              <p>
                KNN is a non-parametric algorithm used for classification and
                regression. It predicts the output based on the majority class
                (classification) or average value (regression) of the nearest neighbors.
              </p>

              <h3>Decision Trees</h3>
              <p>
                Decision Trees split the data into subsets based on feature values,
                creating a tree structure. Each node represents a decision rule, and
                leaves represent outcomes.
              </p>

              <h3>K-Means Clustering</h3>
              <p>
                K-Means is an unsupervised algorithm that partitions data into K
                clusters. It iteratively assigns points to the nearest cluster centroid
                and updates centroids until convergence.
              </p>

              <h3>DBSCAN</h3>
              <p>
                DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
                groups points that are closely packed together while marking outliers as
                noise. It uses density-based criteria for clustering.
              </p>

              <h3>Principal Component Analysis (PCA)</h3>
              <p>
                PCA reduces the dimensionality of data by transforming it into a new set
                of orthogonal components that capture the maximum variance in the data.
              </p>

              <h3>Artificial Neural Networks (ANN)</h3>
              <p>
                ANNs are computational models inspired by the human brain. They consist
                of layers of interconnected nodes (neurons) that process data and learn
                patterns through backpropagation.
              </p>

              <h3>Support Vector Machines (SVM)</h3>
              <p>
                SVMs are supervised learning models that find the optimal hyperplane to
                separate data into classes. They can use different kernels to handle
                linear and non-linear data.
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default Docs;

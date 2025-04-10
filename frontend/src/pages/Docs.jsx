import React, { useState } from 'react';
import { motion } from 'framer-motion';

function Docs() {
  const [expandedSection, setExpandedSection] = useState(null);
  
  const toggleSection = (section) => {
    if (expandedSection === section) {
      setExpandedSection(null);
    } else {
      setExpandedSection(section);
    }
  };

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
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
          }

          h4 {
            font-size: 1.1rem;
            color: #2c3e50;
            margin: 1rem 0 0.5rem 0;
          }

          p {
            font-size: 1rem;
            color: #7f8c8d;
          }

          ol, ul {
            padding-left: 1.5rem;
          }

          ol li, ul li {
            margin-bottom: 0.5rem;
          }

          .collapsible-content {
            overflow: hidden;
            max-height: 0;
            transition: max-height 0.3s ease-out;
          }

          .expanded {
            max-height: 5000px; /* Arbitrarily large value to accommodate all content */
            transition: max-height 0.6s ease-in;
          }

          .math {
            font-style: italic;
          }

          .code-example {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 10px 0;
          }

          .toggle-icon {
            font-size: 1.2rem;
          }
          
          .model-section {
            border-left: 3px solid #3498db;
            padding-left: 15px;
            margin: 15px 0;
          }
          
          .subsection {
            margin-top: 15px;
            margin-bottom: 15px;
          }
        `}
      </style>

      <div className="docs-container">
        <h1 className="model-title">Machine Learning Models Documentation</h1>

        <div className="content-container">
          <div className="input-section">
            <h2 className="section-title">Getting Started</h2>
            <p>
              This documentation will help you understand how to use the ML Visualizer
              application effectively. Choose a model from the home page or navigation
              bar to get started. Each model section provides comprehensive theoretical background and 
              practical implementation details to enhance your understanding.
            </p>
          </div>

          <div className="input-section">
            <h2 className="section-title">How to Use</h2>
            <ol>
              <li>Select a machine learning model from the home page</li>
              <li>Input your data (or load sample data when available)</li>
              <li>Configure any model-specific parameters</li>
              <li>Run the model to see results and visualizations</li>
              <li>Analyze the output metrics and visualization</li>
              <li>Experiment with different parameters to see how they affect the model's performance</li>
            </ol>
          </div>

          <div className="input-section">
            <h2 className="section-title">Model Theory & Implementation</h2>
            <div>
              <div className="model-section">
                <h3 onClick={() => toggleSection('linear-regression')}>
                  Linear Regression
                  <span className="toggle-icon">{expandedSection === 'linear-regression' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'linear-regression' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Linear Regression is a supervised machine learning algorithm used for predicting a continuous target variable
                    based on one or more predictor variables. The core idea is to establish a linear relationship between 
                    input features and output variables.
                  </p>
                  
                  <h4>Mathematical Framework</h4>
                  <p>
                    In simple linear regression (one predictor variable), the relationship is modeled as:
                  </p>
                  <p className="math">y = β₀ + β₁x + ε</p>
                  <p>Where:</p>
                  <ul>
                    <li>y is the dependent variable (target)</li>
                    <li>x is the independent variable (feature)</li>
                    <li>β₀ is the y-intercept (bias)</li>
                    <li>β₁ is the slope coefficient (weight)</li>
                    <li>ε is the error term (residual)</li>
                  </ul>
                  
                  <p>
                    For multiple linear regression (multiple predictor variables), the model expands to:
                  </p>
                  <p className="math">y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε</p>
                  
                  <h4>Parameter Estimation</h4>
                  <p>
                    The key objective is to find the optimal values for the coefficients (β). The most common 
                    method is <strong>Ordinary Least Squares (OLS)</strong>, which minimizes the sum of squared residuals:
                  </p>
                  <p className="math">Minimize: Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (β₀ + β₁x₁ᵢ + ... + βₙxₙᵢ))²</p>
                  
                  <p>
                    For simple linear regression, the closed-form solutions for the parameters are:
                  </p>
                  <p className="math">
                    β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)
                  </p>
                  <p className="math">
                    β₀ = ȳ - β₁x̄
                  </p>
                  
                  <h4>Performance Evaluation</h4>
                  <p>
                    Common metrics for evaluating linear regression models include:
                  </p>
                  <ul>
                    <li><strong>Mean Squared Error (MSE)</strong>: Average of squared differences between predicted and actual values</li>
                    <li><strong>Root Mean Squared Error (RMSE)</strong>: Square root of MSE, in the same units as the target variable</li>
                    <li><strong>R-squared (R²)</strong>: Proportion of variance explained by the model (ranges from 0 to 1)</li>
                    <li><strong>Adjusted R-squared</strong>: R² adjusted for the number of predictors</li>
                  </ul>
                  
                  <h4>Assumptions of Linear Regression</h4>
                  <ol>
                    <li><strong>Linearity</strong>: The relationship between X and Y is linear</li>
                    <li><strong>Independence</strong>: Observations are independent of each other</li>
                    <li><strong>Homoscedasticity</strong>: Constant variance of errors across all levels of X</li>
                    <li><strong>Normality</strong>: Errors are normally distributed</li>
                    <li><strong>No multicollinearity</strong>: Predictor variables are not highly correlated (for multiple regression)</li>
                  </ol>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can visualize how linear regression works by:
                  </p>
                  <ol>
                    <li>Inputting X-Y data points</li>
                    <li>Observing how the best-fit line is calculated</li>
                    <li>Assessing the model's performance through metrics</li>
                    <li>Examining residual plots to validate model assumptions</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Predicting house prices based on size, location, and other features</li>
                    <li>Sales forecasting based on advertising expenditure</li>
                    <li>Understanding relationships between economic indicators</li>
                    <li>Medical research: predicting patient outcomes based on various measurements</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('knn')}>
                  K-Nearest Neighbors (KNN)
                  <span className="toggle-icon">{expandedSection === 'knn' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'knn' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    K-Nearest Neighbors is a non-parametric, instance-based learning algorithm that can be used for 
                    both classification and regression. The core concept is that similar data points exist in close proximity 
                    to each other.
                  </p>
                  
                  <h4>Working Principle</h4>
                  <p>
                    KNN makes predictions based on the similarity of a new instance to training examples. For a new 
                    data point, the algorithm:
                  </p>
                  <ol>
                    <li>Calculates the distance between the new point and all points in the training data</li>
                    <li>Identifies K nearest neighbors (where K is a user-specified parameter)</li>
                    <li>For classification: Returns the majority class among K neighbors</li>
                    <li>For regression: Returns the average value of the K neighbors</li>
                  </ol>
                  
                  <h4>Distance Metrics</h4>
                  <p>
                    The choice of distance metric affects how "nearness" is calculated. Common distance measures include:
                  </p>
                  <ul>
                    <li><strong>Euclidean Distance</strong>: sqrt(Σ(xᵢ - yᵢ)²) — standard straight-line distance</li>
                    <li><strong>Manhattan Distance</strong>: Σ|xᵢ - yᵢ| — sum of absolute differences</li>
                    <li><strong>Minkowski Distance</strong>: (Σ|xᵢ - yᵢ|ᵖ)^(1/p) — generalized distance metric</li>
                    <li><strong>Hamming Distance</strong>: Count of positions where symbols differ (for categorical features)</li>
                  </ul>
                  
                  <h4>Selecting Optimal K</h4>
                  <p>
                    The choice of K significantly impacts model performance:
                  </p>
                  <ul>
                    <li><strong>Small K</strong>: More sensitive to noise, may lead to overfitting</li>
                    <li><strong>Large K</strong>: Smoother decision boundaries, but may miss important patterns</li>
                    <li><strong>Odd K</strong>: Often preferred for binary classification to avoid ties</li>
                  </ul>
                  <p>
                    Cross-validation is typically used to determine the optimal K value for a specific dataset.
                  </p>
                  
                  <h4>Weighted KNN</h4>
                  <p>
                    A refinement to basic KNN is to weight the neighbors' contributions by their distance. Closer neighbors 
                    receive higher weight in the final prediction:
                  </p>
                  <p className="math">
                    Weight = 1/d² or e^(-d)
                  </p>
                  <p>
                    where d is the distance between the query point and the neighbor.
                  </p>
                  
                  <h4>Advantages and Disadvantages</h4>
                  <h5>Advantages:</h5>
                  <ul>
                    <li>Simple to understand and implement</li>
                    <li>No assumptions about data distribution</li>
                    <li>Naturally handles multi-class problems</li>
                    <li>Can model complex decision boundaries</li>
                    <li>No training phase (lazy learning)</li>
                  </ul>
                  
                  <h5>Disadvantages:</h5>
                  <ul>
                    <li>Computationally expensive for large datasets</li>
                    <li>Requires feature scaling</li>
                    <li>Susceptible to the curse of dimensionality</li>
                    <li>Performance degrades with irrelevant features</li>
                    <li>Memory-intensive as all training data must be stored</li>
                  </ul>
                  
                  <h4>Feature Scaling</h4>
                  <p>
                    Feature scaling is crucial for KNN since distance calculations are directly affected by the scale 
                    of each feature. Common scaling methods include:
                  </p>
                  <ul>
                    <li><strong>Min-Max Scaling</strong>: (x - min(x)) / (max(x) - min(x))</li>
                    <li><strong>Z-score Standardization</strong>: (x - mean(x)) / std(x)</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with KNN by:
                  </p>
                  <ol>
                    <li>Selecting between classification or regression mode</li>
                    <li>Inputting data points with class labels or values</li>
                    <li>Adjusting the K parameter to observe changes in decision boundaries</li>
                    <li>Choosing different distance metrics</li>
                    <li>Visualizing how predictions are made based on neighboring points</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Recommendation systems</li>
                    <li>Credit scoring</li>
                    <li>Image recognition</li>
                    <li>Medical diagnosis</li>
                    <li>Pattern recognition</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('decision-trees')}>
                  Decision Trees
                  <span className="toggle-icon">{expandedSection === 'decision-trees' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'decision-trees' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Decision trees are versatile supervised learning algorithms that perform both classification and regression 
                    tasks. They create a model that predicts the value of a target variable by learning simple decision rules 
                    inferred from the data features.
                  </p>
                  
                  <h4>Tree Structure</h4>
                  <p>
                    A decision tree consists of the following components:
                  </p>
                  <ul>
                    <li><strong>Root Node</strong>: Represents the entire dataset and makes the first split</li>
                    <li><strong>Decision Nodes</strong>: Where the data is split based on feature values</li>
                    <li><strong>Leaf/Terminal Nodes</strong>: Final output values (class labels or regression values)</li>
                    <li><strong>Branches</strong>: Connections between nodes representing possible feature values</li>
                  </ul>
                  
                  <h4>Splitting Criteria</h4>
                  <p>
                    The key to building an effective decision tree is determining the best splits at each node. Different 
                    metrics are used based on whether it's a classification or regression tree:
                  </p>
                  
                  <h5>For Classification:</h5>
                  <ul>
                    <li><strong>Gini Impurity</strong>: Measures the probability of incorrect classification
                      <p className="math">Gini = 1 - Σ(pᵢ²)</p>
                      <p>Where pᵢ is the probability of class i</p>
                    </li>
                    <li><strong>Entropy</strong>: Measures the level of disorder or uncertainty
                      <p className="math">Entropy = -Σ(pᵢ × log₂(pᵢ))</p>
                    </li>
                    <li><strong>Information Gain</strong>: Reduction in entropy after a split
                      <p className="math">Gain = Entropy(parent) - Σ((nᵏ/n) × Entropy(child_k))</p>
                    </li>
                  </ul>
                  
                  <h5>For Regression:</h5>
                  <ul>
                    <li><strong>Mean Squared Error (MSE)</strong>: Average of squared differences from the mean
                      <p className="math">MSE = (1/n) × Σ(yᵢ - ȳ)²</p>
                    </li>
                    <li><strong>Mean Absolute Error (MAE)</strong>: Average of absolute differences from the mean
                      <p className="math">MAE = (1/n) × Σ|yᵢ - ȳ|</p>
                    </li>
                    <li><strong>Reduction in Variance</strong>: Measures the reduction in variance after a split</li>
                  </ul>
                  
                  <h4>Tree Growing Algorithm</h4>
                  <p>
                    The general process for constructing a decision tree is:
                  </p>
                  <ol>
                    <li>Select the best feature/threshold to split the data based on splitting criteria</li>
                    <li>Split the data into subsets according to the selected feature</li>
                    <li>Recursively repeat steps 1 and 2 on each subset until stopping criteria are met</li>
                  </ol>
                  
                  <h4>Stopping Criteria (Pruning)</h4>
                  <p>
                    To prevent overfitting, tree growth is limited by:
                  </p>
                  <ul>
                    <li>Maximum tree depth</li>
                    <li>Minimum number of samples required at a leaf node</li>
                    <li>Minimum number of samples required to split a node</li>
                    <li>Minimum decrease in impurity required for a split</li>
                    <li>Post-pruning techniques (removing branches that don't improve generalization)</li>
                  </ul>
                  
                  <h4>Handling Feature Types</h4>
                  <p>
                    Decision trees can natively handle both:
                  </p>
                  <ul>
                    <li><strong>Numerical Features</strong>: Creating binary splits based on thresholds (e.g., Age greater than 30)</li>
                    <li><strong>Categorical Features</strong>: Creating multi-way splits (e.g., Color = Red, Green, Blue)</li>
                  </ul>
                  
                  <h4>Advantages and Disadvantages</h4>
                  <h5>Advantages:</h5>
                  <ul>
                    <li>Simple to understand and interpret</li>
                    <li>Requires little data preprocessing (no normalization needed)</li>
                    <li>Can handle both numerical and categorical data</li>
                    <li>Can model non-linear relationships</li>
                    <li>Inherently performs feature selection</li>
                    <li>Robust to outliers</li>
                  </ul>
                  
                  <h5>Disadvantages:</h5>
                  <ul>
                    <li>Tendency to overfit (especially deep trees)</li>
                    <li>Can be unstable (small changes in data can lead to completely different trees)</li>
                    <li>Biased toward features with more levels when handling categorical data</li>
                    <li>Unable to extrapolate beyond the range of training data</li>
                    <li>Can create biased trees if some classes dominate</li>
                  </ul>
                  
                  <h4>Ensemble Methods</h4>
                  <p>
                    To overcome the limitations of individual decision trees, ensemble methods are often used:
                  </p>
                  <ul>
                    <li><strong>Random Forests</strong>: Builds multiple trees on random subsets of data and features</li>
                    <li><strong>Gradient Boosting</strong>: Builds trees sequentially, where each tree corrects errors from previous trees</li>
                    <li><strong>AdaBoost</strong>: Focuses on difficult-to-classify instances by adjusting weights</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with decision trees by:
                  </p>
                  <ol>
                    <li>Creating a dataset with multiple features</li>
                    <li>Adjusting tree parameters like maximum depth and minimum samples per leaf</li>
                    <li>Visualizing the tree structure and decision boundaries</li>
                    <li>Examining feature importance</li>
                    <li>Testing different splitting criteria</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Customer churn prediction</li>
                    <li>Credit risk assessment</li>
                    <li>Medical diagnosis</li>
                    <li>Spam email detection</li>
                    <li>Recommender systems</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('kmeans')}>
                  K-Means Clustering
                  <span className="toggle-icon">{expandedSection === 'kmeans' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'kmeans' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    K-Means is an unsupervised learning algorithm used for clustering data into K distinct groups based on 
                    feature similarity. The goal is to maximize intra-cluster similarity and minimize inter-cluster similarity.
                  </p>
                  
                  <h4>Algorithm Steps</h4>
                  <ol>
                    <li><strong>Initialization</strong>: Randomly select K points as initial centroids</li>
                    <li><strong>Assignment</strong>: Assign each data point to the nearest centroid, forming K clusters</li>
                    <li><strong>Update</strong>: Recalculate centroids as the mean of all points in each cluster</li>
                    <li><strong>Convergence</strong>: Repeat steps 2-3 until centroids stabilize or maximum iterations reached</li>
                  </ol>
                  
                  <h4>Mathematical Formulation</h4>
                  <p>
                    K-Means aims to minimize the Within-Cluster Sum of Squares (WCSS):
                  </p>
                  <p className="math">
                    WCSS = Σᵏⱼ₌₁ Σᵢ∈Cⱼ ||xᵢ - μⱼ||²
                  </p>
                  <p>Where:</p>
                  <ul>
                    <li>k is the number of clusters</li>
                    <li>Cⱼ is the set of points in cluster j</li>
                    <li>xᵢ is a data point</li>
                    <li>μⱼ is the centroid of cluster j</li>
                  </ul>
                  
                  <h4>Centroid Initialization Techniques</h4>
                  <p>
                    The initial placement of centroids significantly impacts the final clustering result:
                  </p>
                  <ul>
                    <li><strong>Random Initialization</strong>: Select K random data points as initial centroids</li>
                    <li><strong>K-Means++</strong>: Select centroids with probability proportional to their distance from existing centroids</li>
                    <li><strong>Forgy Method</strong>: Randomly assign each point to a cluster, then compute centroids</li>
                    <li><strong>Multiple Restarts</strong>: Run algorithm multiple times with different initializations and select best result</li>
                  </ul>
                  
                  <h4>Determining the Optimal K</h4>
                  <p>
                    Selecting the appropriate number of clusters is a critical challenge in K-Means. Methods include:
                  </p>
                  <ul>
                    <li><strong>Elbow Method</strong>: Plot WCSS against K and look for the "elbow" point where marginal gain decreases</li>
                    <li><strong>Silhouette Analysis</strong>: Measure how similar objects are to their own cluster compared to other clusters</li>
                    <li><strong>Gap Statistic</strong>: Compare the WCSS to its expected value under a null reference distribution</li>
                    <li><strong>Domain Knowledge</strong>: Use prior knowledge about the expected number of natural clusters</li>
                  </ul>
                  
                  <h4>Distance Metrics</h4>
                  <p>
                    While Euclidean distance is most common, other distance metrics can be used:
                  </p>
                  <ul>
                    <li><strong>Euclidean Distance</strong>: sqrt(Σ(xᵢ - yᵢ)²)</li>
                    <li><strong>Manhattan Distance</strong>: Σ|xᵢ - yᵢ|</li>
                    <li><strong>Cosine Similarity</strong>: Measures the cosine of the angle between two vectors (useful for high-dimensional data)</li>
                  </ul>
                  
                  <h4>Limitations and Considerations</h4>
                  <ul>
                    <li><strong>Sensitivity to Initialization</strong>: Different initial centroids can lead to different final clusters</li>
                    <li><strong>Fixed K</strong>: Requires pre-specifying the number of clusters</li>
                    <li><strong>Spherical Clusters</strong>: Assumes clusters are spherical and equally sized</li>
                    <li><strong>Outliers</strong>: Sensitive to outliers which can skew centroid locations</li>
                    <li><strong>Curse of Dimensionality</strong>: Performance degrades in high-dimensional spaces</li>
                    <li><strong>Non-Convex Clusters</strong>: Cannot identify clusters with complex shapes</li>
                  </ul>
                  
                  <h4>Variants and Extensions</h4>
                  <ul>
                    <li><strong>K-Medoids</strong>: Uses actual data points as centers instead of means</li>
                    <li><strong>Fuzzy C-Means</strong>: Allows data points to belong to multiple clusters with varying degrees of membership</li>
                    <li><strong>Mini-Batch K-Means</strong>: Uses small random batches of data for better efficiency on large datasets</li>
                    <li><strong>Kernel K-Means</strong>: Applies kernel functions to handle non-linear cluster boundaries</li>
                  </ul>
                  
                  <h4>Evaluation Metrics</h4>
                  <p>
                    Since clustering is unsupervised, evaluation can be challenging:
                  </p>
                  <ul>
                    <li><strong>Inertia</strong>: Sum of distances of samples to their closest cluster center</li>
                    <li><strong>Silhouette Coefficient</strong>: Measures cohesion and separation of clusters</li>
                    <li><strong>Calinski-Harabasz Index</strong>: Ratio of between-cluster to within-cluster dispersion</li>
                    <li><strong>Davies-Bouldin Index</strong>: Average similarity between each cluster and its most similar cluster</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with K-Means by:
                  </p>
                  <ol>
                    <li>Creating or loading a dataset</li>
                    <li>Setting the number of clusters K</li>
                    <li>Selecting initialization methods</li>
                    <li>Visualizing the iterative clustering process</li>
                    <li>Evaluating cluster quality with various metrics</li>
                    <li>Testing different distance measures</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Customer segmentation</li>
                    <li>Image compression</li>
                    <li>Document clustering</li>
                    <li>Anomaly detection</li>
                    <li>Identifying geographic hotspots</li>
                    <li>Recommender systems</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('dbscan')}>
                  DBSCAN
                  <span className="toggle-icon">{expandedSection === 'dbscan' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'dbscan' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is an unsupervised clustering algorithm 
                    that groups together points that are closely packed in space, while marking points in low-density regions 
                    as outliers or noise.
                  </p>
                  
                  <h4>Core Concepts</h4>
                  <ul>
                    <li><strong>Epsilon (ε)</strong>: The radius that defines the neighborhood distance around a point</li>
                    <li><strong>MinPts</strong>: Minimum number of points required to form a dense region (cluster)</li>
                    <li><strong>Core Point</strong>: A point with at least MinPts points within distance ε</li>
                    <li><strong>Border Point</strong>: A point within distance ε of a core point but with fewer than MinPts neighbors</li>
                    <li><strong>Noise Point</strong>: Any point that is neither a core point nor a border point</li>
                    <li><strong>Directly Density-Reachable</strong>: A point q is directly density-reachable from p if p is a core point and q is within distance ε of p</li>
                    <li><strong>Density-Reachable</strong>: A point q is density-reachable from p if there exists a chain of points p₁, p₂, ..., pₙ where p₁ = p, pₙ = q, and each pᵢ₊₁ is directly density-reachable from pᵢ</li>
                    <li><strong>Density-Connected</strong>: Two points p and q are density-connected if there exists a point o such that both p and q are density-reachable from o</li>
                  </ul>
                  
                  <h4>Algorithm Steps</h4>
                  <ol>
                    <li>For each unvisited point in the dataset:
                      <ul>
                        <li>Mark the point as visited</li>
                        <li>Find all points within distance ε (neighborhood)</li>
                        <li>If the neighborhood contains fewer than MinPts, mark as noise (initially)</li>
                        <li>Otherwise, create a new cluster and add all density-connected points</li>
                      </ul>
                    </li>
                    <li>For each point identified as a core point:
                      <ul>
                        <li>Add all density-reachable points to the same cluster</li>
                        <li>Points previously marked as noise may become border points</li>
                      </ul>
                    </li>
                  </ol>
                  
                  <h4>Parameter Selection</h4>
                  <p>
                    Choosing appropriate parameters is crucial for DBSCAN:
                  </p>
                  <ul>
                    <li><strong>Epsilon (ε)</strong>:
                      <ul>
                        <li>Too small: Many small clusters or noise points</li>
                        <li>Too large: Different clusters may merge inappropriately</li>
                        <li>Heuristic: k-distance graph (plot distance to kth nearest neighbor and look for "elbow")</li>
                      </ul>
                    </li>
                    <li><strong>MinPts</strong>:
                      <ul>
                        <li>Rule of thumb: 2 × dimensions (e.g., 4 for 2D data)</li>
                        <li>Higher for larger datasets or noisy data</li>
                        <li>Lower values make algorithm more sensitive to noise</li>
                      </ul>
                    </li>
                  </ul>
                  
                  <h4>Distance Metrics</h4>
                  <p>
                    DBSCAN can use various distance metrics depending on the data type and domain:
                  </p>
                  <ul>
                    <li>Euclidean distance (most common)</li>
                    <li>Manhattan distance</li>
                    <li>Cosine similarity (for high-dimensional data)</li>
                    <li>Haversine distance (for geographical data)</li>
                    <li>Custom distance functions for specialized applications</li>
                  </ul>
                  
                  <h4>Time and Space Complexity</h4>
                  <ul>
                    <li><strong>Naive implementation</strong>: O(n²) time complexity</li>
                    <li><strong>With spatial indexing</strong> (e.g., R*-tree, k-d tree): O(n log n) average case</li>
                    <li><strong>Space complexity</strong>: O(n) for storing the dataset and visited status</li>
                  </ul>
                  
                  <h4>Advantages and Disadvantages</h4>
                  <h5>Advantages:</h5>
                  <ul>
                    <li>Does not require specifying the number of clusters beforehand</li>
                    <li>Can discover clusters of arbitrary shape (not just spherical)</li>
                    <li>Naturally identifies outliers as noise</li>
                    <li>Only requires two parameters</li>
                    <li>Can handle clusters of different sizes and densities (to some extent)</li>
                  </ul>
                  
                  <h5>Disadvantages:</h5>
                  <ul>
                    <li>Struggles with varying density clusters</li>
                    <li>Sensitive to parameter selection</li>
                    <li>Computationally expensive for large datasets</li>
                    <li>Does not work well with high-dimensional data due to "curse of dimensionality"</li>
                    <li>Cannot handle datasets with large differences in densities</li>
                  </ul>
                  
                  <h4>Variants and Extensions</h4>
                  <ul>
                    <li><strong>OPTICS</strong>: Ordering points to identify the clustering structure; addresses varying density</li>
                    <li><strong>HDBSCAN</strong>: Hierarchical DBSCAN; eliminates ε parameter and creates a cluster hierarchy</li>
                    <li><strong>ST-DBSCAN</strong>: Spatio-temporal extension for clustering space-time data</li>
                    <li><strong>DBSCAN++</strong>: More efficient approximation of DBSCAN</li>
                    <li><strong>Parallel and distributed implementations</strong> for big data</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with DBSCAN by:
                  </p>
                  <ol>
                    <li>Creating or loading a dataset with varied shapes and densities</li>
                    <li>Adjusting epsilon and MinPts parameters</li>
                    <li>Visualizing core, border, and noise points</li>
                    <li>Comparing results with K-means clustering</li>
                    <li>Analyzing how clusters evolve with different parameter settings</li>
                    <li>Observing how the algorithm handles noise and outliers</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Anomaly detection in security and fraud prevention</li>
                    <li>Spatial data analysis in GIS applications</li>
                    <li>Traffic pattern analysis</li>
                    <li>Image segmentation</li>
                    <li>Disease outbreak detection</li>
                    <li>Social network analysis</li>
                    <li>Biodiversity and species habitat studies</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('pca')}>
                  Principal Component Analysis (PCA)
                  <span className="toggle-icon">{expandedSection === 'pca' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'pca' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional 
                    data into a lower-dimensional space while preserving as much variance as possible. It works by finding 
                    orthogonal axes (principal components) along which data varies the most.
                  </p>
                  
                  <h4>Mathematical Formulation</h4>
                  <p>
                    PCA uses linear algebra concepts to transform data:
                  </p>
                  <ol>
                    <li><strong>Standardization</strong>: Center the data by subtracting the mean and optionally scale by dividing by the standard deviation</li>
                    <li><strong>Covariance Matrix</strong>: Compute the covariance matrix of the standardized data
                      <p className="math">C = (1/n) × X^T × X</p>
                      <p>where X is the centered data matrix and n is the number of samples</p>
                    </li>
                    <li><strong>Eigendecomposition</strong>: Compute the eigenvectors and eigenvalues of the covariance matrix</li>
                    <li><strong>Principal Components</strong>: Eigenvectors represent the principal components, and eigenvalues indicate the amount of variance explained</li>
                    <li><strong>Projection</strong>: Project the original data onto the principal components
                      <p className="math">Z = X × W</p>
                      <p>where W is the matrix of eigenvectors</p>
                    </li>
                  </ol>
                  
                  <h4>Eigenvalues and Explained Variance</h4>
                  <p>
                    Each eigenvalue λᵢ corresponds to the variance captured by the ith principal component. The proportion of 
                    variance explained by each component is:
                  </p>
                  <p className="math">
                    Explained Variance Ratio = λᵢ / Σλⱼ
                  </p>
                  <p>
                    Cumulative explained variance can be used to determine how many components to retain.
                  </p>
                  
                  <h4>Selecting the Number of Components</h4>
                  <p>
                    Several methods can guide the choice of how many principal components to keep:
                  </p>
                  <ul>
                    <li><strong>Explained Variance Threshold</strong>: Keep components that cumulatively explain a certain percentage of variance (e.g., 95%)</li>
                    <li><strong>Kaiser's Rule</strong>: Keep components with eigenvalues greater than 1</li>
                    <li><strong>Scree Plot</strong>: Plot eigenvalues and look for the "elbow" where the curve flattens</li>
                    <li><strong>Parallel Analysis</strong>: Compare eigenvalues to those obtained from random data</li>
                  </ul>
                  
                  <h4>SVD Approach</h4>
                  <p>
                    In practice, PCA is often implemented using Singular Value Decomposition (SVD):
                  </p>
                  <p className="math">
                    X = U × Σ × V^T
                  </p>
                  <p>
                    Where:
                  </p>
                  <ul>
                    <li>U contains the left singular vectors</li>
                    <li>Σ is a diagonal matrix containing the singular values</li>
                    <li>V contains the right singular vectors (equivalent to eigenvectors of X^T × X)</li>
                  </ul>
                  <p>
                    The principal components are the columns of V, and the singular values (squared) are proportional to the eigenvalues.
                  </p>
                  
                  <h4>Applications of PCA</h4>
                  <p>
                    PCA serves multiple purposes in data analysis:
                  </p>
                  <ul>
                    <li><strong>Dimensionality Reduction</strong>: Compress data while preserving important patterns</li>
                    <li><strong>Visualization</strong>: Visualize high-dimensional data in 2D or 3D</li>
                    <li><strong>Noise Reduction</strong>: Lower-order components often capture noise, which can be discarded</li>
                    <li><strong>Feature Extraction</strong>: Create uncorrelated features for downstream machine learning tasks</li>
                    <li><strong>Multicollinearity Handling</strong>: Transform correlated features into uncorrelated components</li>
                  </ul>
                  
                  <h4>Data Preprocessing for PCA</h4>
                  <p>
                    Proper preprocessing is essential for PCA:
                  </p>
                  <ul>
                    <li><strong>Centering</strong>: Always subtract the mean (required for PCA)</li>
                    <li><strong>Scaling</strong>: Usually advisable to standardize features (especially when features have different units)</li>
                    <li><strong>Outlier Treatment</strong>: PCA is sensitive to outliers, which may need to be addressed</li>
                    <li><strong>Missing Value Handling</strong>: Impute missing values before applying PCA</li>
                  </ul>
                  
                  <h4>Limitations and Considerations</h4>
                  <ul>
                    <li>Only captures linear relationships between variables</li>
                    <li>Sensitive to feature scaling</li>
                    <li>May not preserve distances between data points</li>
                    <li>Principal components may be difficult to interpret</li>
                    <li>Not suitable when non-linear patterns are important</li>
                    <li>Affected by outliers</li>
                  </ul>
                  
                  <h4>Variants and Extensions</h4>
                  <ul>
                    <li><strong>Kernel PCA</strong>: Uses kernel methods to handle non-linear relationships</li>
                    <li><strong>Sparse PCA</strong>: Adds sparsity constraints to improve interpretability</li>
                    <li><strong>Incremental PCA</strong>: Processes data in batches for large datasets</li>
                    <li><strong>Robust PCA</strong>: Less sensitive to outliers</li>
                    <li><strong>Probabilistic PCA</strong>: Adds a probabilistic framework to PCA</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with PCA by:
                  </p>
                  <ol>
                    <li>Loading or creating multi-dimensional data</li>
                    <li>Standardizing features</li>
                    <li>Applying PCA and visualizing the transformed data</li>
                    <li>Examining explained variance and scree plots</li>
                    <li>Reconstructing data from selected principal components</li>
                    <li>Comparing original and reduced-dimension representations</li>
                    <li>Visualizing eigenvectors (principal components) as directions in the original feature space</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Image compression and facial recognition</li>
                    <li>Gene expression analysis in bioinformatics</li>
                    <li>Financial data analysis and risk management</li>
                    <li>Signal processing and noise reduction</li>
                    <li>Recommendation systems</li>
                    <li>Meteorological data analysis</li>
                    <li>Preprocessing step for many machine learning pipelines</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('ann')}>
                  Artificial Neural Networks (ANN)
                  <span className="toggle-icon">{expandedSection === 'ann' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'ann' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks in animal 
                    brains. They consist of interconnected nodes (neurons) organized in layers that process and transform input 
                    data to produce desired outputs.
                  </p>
                  
                  <h4>Basic Structure</h4>
                  <ul>
                    <li><strong>Input Layer</strong>: Receives the initial data (features)</li>
                    <li><strong>Hidden Layers</strong>: Intermediate layers where most computation occurs</li>
                    <li><strong>Output Layer</strong>: Produces the final prediction or classification</li>
                    <li><strong>Neurons</strong>: Basic units that receive inputs, apply weights and activations, and pass outputs forward</li>
                    <li><strong>Weights</strong>: Parameters that determine the strength of connections between neurons</li>
                    <li><strong>Biases</strong>: Additional parameters that allow shifting the activation function</li>
                  </ul>
                  
                  <h4>Mathematical Model of a Neuron</h4>
                  <p>
                    Each neuron processes inputs through the following steps:
                  </p>
                  <ol>
                    <li><strong>Linear Combination</strong>: Calculate weighted sum of inputs plus bias
                      <p className="math">z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b</p>
                    </li>
                    <li><strong>Activation Function</strong>: Apply non-linear transformation
                      <p className="math">a = f(z)</p>
                    </li>
                  </ol>
                  
                  <h4>Common Activation Functions</h4>
                  <ul>
                    <li><strong>Sigmoid</strong>: f(x) = 1/(1+e^(-x))
                      <ul>
                        <li>Range: (0, 1)</li>
                        <li>Historically popular, but suffers from vanishing gradient</li>
                        <li>Used in binary classification output layers</li>
                      </ul>
                    </li>
                    <li><strong>Tanh</strong>: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
                      <ul>
                        <li>Range: (-1, 1)</li>
                        <li>Zero-centered outputs</li>
                        <li>Still suffers from vanishing gradient</li>
                      </ul>
                    </li>
                    <li><strong>ReLU</strong> (Rectified Linear Unit): f(x) = max(0, x)
                      <ul>
                        <li>Range: [0, ∞)</li>
                        <li>Computationally efficient</li>
                        <li>Helps mitigate vanishing gradient</li>
                        <li>May suffer from "dying ReLU" problem</li>
                      </ul>
                    </li>
                    <li><strong>Leaky ReLU</strong>: f(x) = max(αx, x), where α is a small constant
                      <ul>
                        <li>Addresses "dying ReLU" problem</li>
                      </ul>
                    </li>
                    <li><strong>Softmax</strong>: f(x)ᵢ = e^xᵢ / Σⱼe^xⱼ
                      <ul>
                        <li>Used in multi-class classification output layers</li>
                        <li>Outputs sum to 1, can be interpreted as probabilities</li>
                      </ul>
                    </li>
                  </ul>
                  
                  <h4>Feedforward and Backpropagation</h4>
                  <p>
                    The training process consists of two main phases:
                  </p>
                  <ol>
                    <li><strong>Feedforward</strong>: Input data flows through the network to generate predictions
                      <ul>
                        <li>Calculate activations layer by layer from input to output</li>
                        <li>Compare output with target values to calculate loss</li>
                      </ul>
                    </li>
                    <li><strong>Backpropagation</strong>: Error propagates backward to update weights
                      <ul>
                        <li>Calculate gradients of the loss with respect to weights and biases</li>
                        <li>Apply chain rule to propagate error gradients backward</li>
                        <li>Update parameters using gradient descent or variants</li>
                      </ul>
                    </li>
                  </ol>
                  
                  <h4>Loss Functions</h4>
                  <p>
                    Different problems require different loss functions:
                  </p>
                  <ul>
                    <li><strong>Mean Squared Error (MSE)</strong>: For regression problems
                      <p className="math">MSE = (1/n) × Σ(yᵢ - ŷᵢ)²</p>
                    </li>
                    <li><strong>Binary Cross-Entropy</strong>: For binary classification
                      <p className="math">BCE = -(1/n) × Σ(yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ))</p>
                    </li>
                    <li><strong>Categorical Cross-Entropy</strong>: For multi-class classification
                      <p className="math">CCE = -(1/n) × Σᵢ Σⱼ yᵢⱼ × log(ŷᵢⱼ)</p>
                    </li>
                    <li><strong>Hinge Loss</strong>: Used for maximum-margin classification
                      <p className="math">Hinge = (1/n) × Σ max(0, 1 - yᵢ × ŷᵢ)</p>
                    </li>
                  </ul>
                  
                  <h4>Optimization Algorithms</h4>
                  <ul>
                    <li><strong>Gradient Descent</strong>: Basic algorithm that updates weights in the opposite direction of gradients</li>
                    <li><strong>Stochastic Gradient Descent (SGD)</strong>: Updates parameters using one sample at a time</li>
                    <li><strong>Mini-Batch SGD</strong>: Updates parameters using small batches of samples</li>
                    <li><strong>Momentum</strong>: Adds a fraction of previous update to current update</li>
                    <li><strong>RMSprop</strong>: Adapts learning rates based on recent gradient magnitudes</li>
                    <li><strong>Adam</strong>: Combines ideas from momentum and RMSprop</li>
                  </ul>
                  
                  <h4>Regularization Techniques</h4>
                  <p>
                    Preventing overfitting is crucial for neural networks:
                  </p>
                  <ul>
                    <li><strong>L1/L2 Regularization</strong>: Adds penalty terms to the loss function based on weight magnitudes</li>
                    <li><strong>Dropout</strong>: Randomly sets a fraction of neurons to zero during training</li>
                    <li><strong>Batch Normalization</strong>: Normalizes activations within each mini-batch</li>
                    <li><strong>Early Stopping</strong>: Halts training when validation performance starts to degrade</li>
                    <li><strong>Data Augmentation</strong>: Creates additional training samples through transformations</li>
                  </ul>
                  
                  <h4>Architectural Considerations</h4>
                  <ul>
                    <li><strong>Number of Layers</strong>: More layers can capture more complex patterns but are harder to train</li>
                    <li><strong>Neurons per Layer</strong>: More neurons increase model capacity but may lead to overfitting</li>
                    <li><strong>Skip Connections</strong>: Allow information to bypass layers (as in ResNet)</li>
                    <li><strong>Transfer Learning</strong>: Reusing pre-trained network layers for new tasks</li>
                    <li><strong>Ensembling</strong>: Combining multiple networks to improve performance</li>
                  </ul>
                  
                  <h4>Types of Neural Networks</h4>
                  <ul>
                    <li><strong>Multilayer Perceptron (MLP)</strong>: Basic feedforward network with fully connected layers</li>
                    <li><strong>Convolutional Neural Network (CNN)</strong>: Specializes in grid-like data (images)</li>
                    <li><strong>Recurrent Neural Network (RNN)</strong>: Processes sequential data with memory capabilities</li>
                    <li><strong>Long Short-Term Memory (LSTM)</strong>: Advanced RNN that addresses vanishing gradient problem</li>
                    <li><strong>Autoencoders</strong>: Encode data into lower dimensions and then reconstruct it</li>
                    <li><strong>Generative Adversarial Networks (GANs)</strong>: Two networks compete to generate realistic data</li>
                  </ul>
                  
                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with neural networks by:
                  </p>
                  <ol>
                    <li>Designing network architecture (layers, neurons, activations)</li>
                    <li>Configuring training parameters (learning rate, batch size, epochs)</li>
                    <li>Visualizing training progress (loss, accuracy curves)</li>
                    <li>Observing how neurons activate for different inputs</li>
                    <li>Testing different regularization techniques</li>
                    <li>Visualizing decision boundaries formed by the network</li>
                  </ol>
                  
                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Image and speech recognition</li>
                    <li>Natural language processing</li>
                    <li>Medical diagnosis and healthcare</li>
                    <li>Financial forecasting and algorithmic trading</li>
                    <li>Autonomous vehicles</li>
                    <li>Recommendation systems</li>
                    <li>Game playing and strategic decision making</li>
                  </ul>
                </div>
              </div>

              <div className="model-section">
                <h3 onClick={() => toggleSection('svm')}>
                  Support Vector Machines (SVM)
                  <span className="toggle-icon">{expandedSection === 'svm' ? '▼' : '►'}</span>
                </h3>
                <div className={`collapsible-content ${expandedSection === 'svm' ? 'expanded' : ''}`}>
                  <h4>Theoretical Foundation</h4>
                  <p>
                    Support Vector Machines (SVMs) are powerful supervised learning models used for classification, regression, 
                    and outlier detection. The key idea is to find the optimal hyperplane that maximizes the margin between 
                    different classes in the feature space.
                  </p>
                  
                  <h4>Linear SVM for Classification</h4>
                  <p>
                    For linearly separable data, SVM finds a hyperplane defined by:
                  </p>
                  <p className="math">
                    w·x + b = 0
                  </p>
                  <p>Where:</p>
                  <ul>
                    <li>w is the weight vector (normal to the hyperplane)</li>
                    <li>x is the feature vector</li>
                    <li>b is the bias term</li>
                  </ul>
                  <p>
                    The decision function for classification is:
                  </p>
                  <p className="math">
                    f(x) = sign(w·x + b)
                  </p>
                  
                  <h4>Margin Maximization</h4>
                  <p>
                    SVM aims to maximize the margin between the hyperplane and the closest data points (support vectors). 
                    This can be formulated as an optimization problem:
                  </p>
                  <p className="math">
                    Minimize ||w||² / 2
                  </p>
                  <p className="math">
                    Subject to: y_i(w·x_i + b) ≥ 1 for all i
                  </p>
                  <p>
                    Where yᵢ ∈ -1,1 is the class label for the ith sample.
                  </p>
                  <p>
                    The margin width equals 2/||w||, so minimizing ||w||² maximizes the margin.
                  </p>
                  
                  <h4>Soft Margin SVM</h4>
                  <p>
                    In practice, data may not be perfectly separable. The soft margin approach introduces slack variables ξᵢ 
                    to allow some misclassifications:
                  </p>
                  <p className="math">
                    Minimize ||w||² / 2 + C × Σξᵢ
                  </p>
                  <p className="math">
                    Subject to: y_i(w·x_i + b) ≥ 1 - ξᵢ and ξᵢ ≥ 0 for all i
                  </p>
                  <p>
                    The parameter C controls the trade-off between maximizing the margin and minimizing classification error:
                  </p>
                  <ul>
                    <li>Large C: Focus on minimizing misclassifications (may lead to overfitting)</li>
                    <li>Small C: Focus on maximizing the margin (may allow more misclassifications)</li>
                  </ul>
                  
                  <h4>The Kernel Trick</h4>
                  <p>
                    For non-linearly separable data, SVM uses the "kernel trick" to implicitly map data to a higher-dimensional 
                    space where it becomes linearly separable. This is done by replacing dot products with kernel functions:
                  </p>
                  <p className="math">
                    K(x_i, x_j) = φ(x_i)·φ(x_j)
                  </p>
                  <p>
                    Common kernel functions include:
                  </p>
                  <ul>
                    <li><strong>Linear Kernel</strong>: K(x_i, x_j) = x_i·x_j</li>
                    <li><strong>Polynomial Kernel</strong>: K(x_i, x_j) = (γ × x_i·x_j + r)^d
                      <ul>
                        <li>γ, r are parameters</li>
                        <li>d is the polynomial degree</li>
                      </ul>
                    </li>
                    <li><strong>Radial Basis Function (RBF)/Gaussian Kernel</strong>: K(x_i, x_j) = exp(-γ||x_i - x_j||²)
                      <ul>
                        <li>γ controls the influence radius of each support vector</li>
                      </ul>
                    </li>
                    <li><strong>Sigmoid Kernel</strong>: K(x_i, x_j) = tanh(γ × x_i·x_j + r)</li>
                  </ul>
                  
                  <h4>Support Vectors</h4>
                  <p>
                    Support vectors are the data points closest to the decision boundary that influence its position and orientation. 
                    They are the most difficult samples to classify and have the following properties:
                  </p>
                  <ul>
                    <li>Lie on the margin boundaries or within the margin (in soft margin case)</li>
                    <li>Have non-zero Lagrange multipliers in the dual formulation</li>
                    <li>Often represent a small subset of the training data</li>
                    <li>Completely define the decision boundary (other points can be removed without changing it)</li>
                  </ul>
                  
                  <h4>Multi-class Classification</h4>
                  <p>
                    SVMs are inherently binary classifiers, but can be extended to multi-class problems using:
                  </p>
                  <ul>
                    <li><strong>One-vs-Rest (OvR)</strong>: Train N binary classifiers, each separating one class from all others
                      <ul>
                        <li>For each class i, fit an SVM with positive examples from class i and negative examples from all other classes</li>
                        <li>During prediction, the class with the highest decision function value is chosen</li>
                      </ul>
                    </li>
                    <li><strong>One-vs-One (OvO)</strong>: Train N(N-1)/2 binary classifiers, one for each pair of classes
                      <ul>
                        <li>Each classifier distinguishes between a pair of classes</li>
                        <li>During prediction, each classifier votes for a class, and the class with most votes wins</li>
                        <li>Often preferred for SVMs as it trains on smaller subproblems</li>
                      </ul>
                    </li>
                    <li><strong>Error-Correcting Output Codes (ECOC)</strong>: Advanced approach that leverages coding theory to increase robustness</li>
                  </ul>

                  <h4>SVM for Regression (SVR)</h4>
                  <p>
                    Support Vector Regression applies SVM principles to predict continuous values by:
                  </p>
                  <ul>
                    <li>Introducing a margin of tolerance (epsilon) around the regression line</li>
                    <li>Penalizing only predictions with errors greater than epsilon</li>
                    <li>Formulating the optimization problem to find the flattest tube with radius epsilon that contains most training data</li>
                  </ul>
                  <p className="math">
                    Minimize ||w||²/2 + C × Σ(ξᵢ + ξᵢ*)
                  </p>
                  <p className="math">
                    Subject to: yᵢ - (w·xᵢ + b) ≤ ε + ξᵢ and (w·xᵢ + b) - yᵢ ≤ ε + ξᵢ* and ξᵢ, ξᵢ* ≥ 0 for all i
                  </p>
                  <p>
                    Where ξᵢ and ξᵢ* are slack variables for exceeding the target value by more than ε and falling below the target value by more than ε, respectively.
                  </p>

                  <h4>Hyperparameter Tuning</h4>
                  <p>
                    Key hyperparameters that affect SVM performance include:
                  </p>
                  <ul>
                    <li><strong>C parameter</strong>: Controls the trade-off between maximizing margin and minimizing classification errors</li>
                    <li><strong>Kernel choice</strong>: Depends on the data's structure and separability</li>
                    <li><strong>Kernel-specific parameters</strong>: Such as γ for RBF kernel, which controls the influence radius of support vectors</li>
                    <li><strong>Epsilon (ε)</strong>: For SVR, defines the width of the insensitive tube</li>
                  </ul>
                  <p>
                    Grid search with cross-validation is commonly used to find optimal hyperparameter values.
                  </p>

                  <h4>Computational Complexity</h4>
                  <ul>
                    <li><strong>Training</strong>: O(n²) to O(n³) time complexity, where n is the number of training samples</li>
                    <li><strong>Prediction</strong>: O(nsv × d), where nsv is the number of support vectors and d is the number of features</li>
                    <li><strong>Space complexity</strong>: O(nsv × d) as only support vectors need to be stored</li>
                  </ul>

                  <h4>Advantages and Disadvantages</h4>
                  <h5>Advantages:</h5>
                  <ul>
                    <li>Effective in high-dimensional spaces</li>
                    <li>Memory efficient as only support vectors need to be stored</li>
                    <li>Robust against overfitting, especially in high-dimensional spaces</li>
                    <li>Versatile due to different kernel options</li>
                    <li>Well-founded in mathematical theory</li>
                  </ul>

                  <h5>Disadvantages:</h5>
                  <ul>
                    <li>Doesn't directly provide probability estimates (requires additional calibration)</li>
                    <li>Computationally intensive for large datasets</li>
                    <li>Sensitive to noise</li>
                    <li>Requires careful feature scaling</li>
                    <li>Kernel and hyperparameter selection can be challenging</li>
                  </ul>

                  <h4>Feature Scaling</h4>
                  <p>
                    Feature scaling is crucial for SVMs as:
                  </p>
                  <ul>
                    <li>Features with larger scales can dominate the calculation of distances</li>
                    <li>Kernel functions like RBF are sensitive to feature magnitudes</li>
                    <li>Common scaling methods include:
                      <ul>
                        <li>Min-Max Scaling: (x - min(x)) / (max(x) - min(x))</li>
                        <li>Standardization: (x - mean(x)) / std(x)</li>
                      </ul>
                    </li>
                  </ul>

                  <h4>SVM vs. Other Classifiers</h4>
                  <ul>
                    <li>Compared to logistic regression, SVMs don't provide direct probability outputs</li>
                    <li>Unlike decision trees, SVMs handle high-dimensional data efficiently</li>
                    <li>SVMs often outperform other algorithms when there's a clear margin of separation</li>
                    <li>Neural networks may perform better with large amounts of data and complex patterns</li>
                    <li>SVMs typically require less data than deep learning approaches</li>
                  </ul>

                  <h4>Practical Implementation</h4>
                  <p>
                    In our simulator, you can experiment with SVMs by:
                  </p>
                  <ol>
                    <li>Creating or loading a dataset</li>
                    <li>Selecting a kernel function (linear, polynomial, RBF)</li>
                    <li>Adjusting the C parameter and kernel-specific parameters</li>
                    <li>Visualizing the decision boundary and support vectors</li>
                    <li>Evaluating the model's performance with metrics like accuracy, precision, recall</li>
                    <li>Comparing results with different kernels and parameters</li>
                  </ol>

                  <h4>Real-world Applications</h4>
                  <ul>
                    <li>Text and document classification</li>
                    <li>Image classification and face detection</li>
                    <li>Handwriting recognition</li>
                    <li>Bioinformatics (protein classification, cancer classification)</li>
                    <li>Financial analysis and fraud detection</li>
                    <li>Remote sensing and hyperspectral image analysis</li>
                  </ul>

                  <h4>Advanced Topics</h4>
                  <ul>
                    <li><strong>Online SVMs</strong>: For incremental learning with streaming data</li>
                    <li><strong>Structured SVMs</strong>: For predicting structured outputs like sequences or trees</li>
                    <li><strong>Multiple Kernel Learning</strong>: Combines different kernels to enhance performance</li>
                    <li><strong>One-Class SVM</strong>: For novelty or anomaly detection by learning the boundary of normal data</li>
                    <li><strong>Transductive SVM</strong>: Incorporates unlabeled data in the training process</li>
                  </ul>
                </div>
              </div>

            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
 
export default Docs;


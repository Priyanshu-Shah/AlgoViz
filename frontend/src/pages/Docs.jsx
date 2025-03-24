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
      <h1 className="model-title">Documentation</h1>
      
      <div className="content-container" style={{ gridTemplateColumns: '1fr' }}>
        <div className="input-section">
          <h2 className="section-title">Getting Started</h2>
          <p style={{ marginBottom: '1.5rem', lineHeight: '1.6' }}>
            This documentation will help you understand how to use the ML Visualizer 
            application effectively. Choose a model from the home page or navigation 
            bar to get started.
          </p>
          
          <h2 className="section-title">Available Models</h2>
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              <Link to="/lin-reg">Linear Regression</Link>
            </h3>
            <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>
              Input pairs of X and Y values to visualize a linear relationship between variables. 
              The model will calculate the best-fitting line and various metrics.
            </p>
            
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              <Link to="/knn">K-Nearest Neighbors (KNN)</Link>
            </h3>
            <p style={{ lineHeight: '1.6' }}>
              Choose between classification or regression. For classification, input points with class labels. 
              For regression, input X-Y pairs. The model will make predictions based on the nearest neighbors.
            </p>
          </div>
          
          <h2 className="section-title">How to Use</h2>
          <ol style={{ lineHeight: '1.6', marginLeft: '1.5rem' }}>
            <li>Select a machine learning model from the home page</li>
            <li>Input your data (or load sample data when available)</li>
            <li>Configure any model-specific parameters</li>
            <li>Run the model to see results and visualizations</li>
            <li>Analyze the output metrics and visualization</li>
          </ol>
        </div>
      </div>
    </motion.div>
  );
}

export default Docs;

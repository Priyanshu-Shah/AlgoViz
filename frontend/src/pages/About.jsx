import React from 'react';
import { motion } from 'framer-motion';

function About() {
  return (
    <motion.div 
      className="model-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="model-title">About ML Visualizer</h1>
      
      <div className="content-container" style={{ gridTemplateColumns: '1fr' }}>
        <div className="input-section">
          <p style={{ marginBottom: '1.5rem', lineHeight: '1.6' }}>
            The ML Visualizer is an educational tool designed to help users understand machine learning 
            concepts through interactive visualizations. This project allows you to experiment with 
            different algorithms and see how they work in real-time.
          </p>
          
          <h2 className="section-title">Project Goals</h2>
          <ul style={{ lineHeight: '1.6', marginLeft: '1.5rem', marginBottom: '1.5rem' }}>
            <li>Make machine learning concepts more accessible</li>
            <li>Provide interactive visualizations for better understanding</li>
            <li>Offer a platform for experimentation with different algorithms</li>
            <li>Demonstrate the practical applications of machine learning models</li>
          </ul>
          
          <h2 className="section-title">Technologies Used</h2>
          <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>
            This project is built using a combination of modern web and data science technologies:
          </p>
          <ul style={{ lineHeight: '1.6', marginLeft: '1.5rem' }}>
            <li><strong>Frontend:</strong> React, Framer Motion, React Router</li>
            <li><strong>Backend:</strong> Flask, Python</li>
            <li><strong>Machine Learning:</strong> scikit-learn, NumPy, Pandas</li>
            <li><strong>Visualization:</strong> Matplotlib, seaborn</li>
          </ul>
        </div>
      </div>
    </motion.div>
  );
}

export default About;

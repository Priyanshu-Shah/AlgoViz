import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './HomePage.css';

function HomePage() {
  const navigate = useNavigate();
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 80 }
    }
  };

  // Model definitions
  const supervisedModels = [
    {
      name: "Polynomial Regression",
      description: "Fit polynomial curves to predict continuous values",
      path: "/reg",
      image: "https://cdn.iconscout.com/icon/premium/png-256-thumb/linear-regression-1522289-1289191.png",
      implemented: true
    },
    {
      name: "K-Nearest Neighbors",
      description: "Classify or regress based on closest training examples",
      path: "/knn",
      image: "/logo/knn.png",
      implemented: true
    },
    {
      name: "Decision Tree",
      description: "Make decisions based on hierarchical feature splits",
      path: "/d-trees",
      image: "https://static.thenounproject.com/png/1119933-200.png",
      implemented: true
    },
    {
      name: "Support Vector Machine",
      description: "Classify data by finding optimal hyperplanes",
      path: "/svm",
      image: "https://cdn.iconscout.com/icon/premium/png-256-thumb/support-vector-machines-8295140-6881761.png?f=webp&w=256",
      implemented: true
    },
    {
      name: "Artificial Neural Network",
      description: "Deep learning model inspired by the human brain",
      path: "/ann",
      image: "https://w7.pngwing.com/pngs/613/402/png-transparent-black-dot-with-lines-illustration-artificial-neural-network-deep-learning-machine-learning-artificial-intelligence-computer-network-others-miscellaneous-angle-symmetry.png",
      implemented: true
    }
  ];

  const unsupervisedModels = [
    {
      name: "K-Means Clustering",
      description: "Group data points into k distinct clusters",
      path: "/kmeans",
      image: "https://static.thenounproject.com/png/961658-200.png",
      implemented: true
    },
    {
      name: "DBSCAN",
      description: "Density-based spatial clustering of applications with noise",
      path: "/DBScan",
      image: "/logo/dbscan.png",
      implemented: true
    },
    {
      name: "Principal Component Analysis",
      description: "Reduce dimensionality while preserving variance",
      path: "/pca",
      image: "/logo/pca.png",
      implemented: true
    }
  ];

  return (
    <div className="homepage">
      {/* Hero Section */}
      <motion.div 
        className="hero-section"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <h1 className="homepage-title">Machine Learning Models</h1>
        <p className="homepage-subtitle">Explore interactive algorithms and visualize data patterns</p>
        <motion.div 
          className="decorative-line"
          initial={{ width: 0 }}
          animate={{ width: "150px" }}
          transition={{ duration: 1, delay: 0.5 }}
        ></motion.div>
      </motion.div>

      {/* Supervised Learning Section */}
      <motion.div 
        className="learning-section"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <h2 className="section-heading">Supervised Learning</h2>
        <div className="model-grid">
          {supervisedModels.map((model, index) => (
            <motion.div
              key={index}
              className={`model-card ${!model.implemented ? 'disabled' : ''}`}
              onClick={() => model.implemented && navigate(model.path)}
              variants={itemVariants}
              whileHover={model.implemented ? { 
                scale: 1.05, 
                boxShadow: "0px 10px 20px rgba(0,0,0,0.15)" 
              } : {}}
              whileTap={model.implemented ? { scale: 0.98 } : {}}
            >
              <div className="card-content">
                <div className="card-image-container">
                  <motion.img
                    src={model.image}
                    alt={model.name}
                    className="model-image"
                    whileHover={model.implemented ? { rotate: 5 } : {}}
                    transition={{ type: "spring", stiffness: 200 }}
                  />
                </div>
                <h3 className="model-name">{model.name}</h3>
                <p className="model-description">{model.description}</p>
                <div className="card-cta">
                  {model.implemented ? (
                    <>
                      <span>Explore</span>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M12 5L19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    </>
                  ) : (
                    <span className="coming-soon">Coming Soon</span>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Unsupervised Learning Section */}
      <motion.div 
        className="learning-section"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <h2 className="section-heading">Unsupervised Learning</h2>
        <div className="model-grid">
          {unsupervisedModels.map((model, index) => (
            <motion.div
              key={index}
              className={`model-card ${!model.implemented ? 'disabled' : ''}`}
              onClick={() => model.implemented && navigate(model.path)}
              variants={itemVariants}
              whileHover={model.implemented ? { 
                scale: 1.05, 
                boxShadow: "0px 10px 20px rgba(0,0,0,0.15)" 
              } : {}}
              whileTap={model.implemented ? { scale: 0.98 } : {}}
            >
              <div className="card-content">
                <div className="card-image-container">
                  <motion.img
                    src={model.image}
                    alt={model.name}
                    className="model-image"
                    whileHover={model.implemented ? { rotate: 5 } : {}}
                    transition={{ type: "spring", stiffness: 200 }}
                  />
                </div>
                <h3 className="model-name">{model.name}</h3>
                <p className="model-description">{model.description}</p>
                <div className="card-cta">
                  {model.implemented ? (
                    <>
                      <span>Explore</span>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M12 5L19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    </>
                  ) : (
                    <span className="coming-soon">Coming Soon</span>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
      
      {/* Background effects */}
      <div className="background-decoration">
        <motion.div 
          className="bg-circle circle-1"
          animate={{ 
            y: [0, -15, 0], 
            opacity: [0.5, 0.7, 0.5] 
          }}
          transition={{ 
            repeat: Infinity, 
            duration: 8, 
            ease: "easeInOut" 
          }}
        ></motion.div>
        <motion.div 
          className="bg-circle circle-2"
          animate={{ 
            y: [0, 20, 0], 
            opacity: [0.3, 0.6, 0.3] 
          }}
          transition={{ 
            repeat: Infinity, 
            duration: 10, 
            ease: "easeInOut" 
          }}
        ></motion.div>
      </div>
    </div>
  );
}

export default HomePage;

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
      name: "Linear Regression",
      description: "Predict continuous values based on linear relationships",
      path: "/lin-reg",
      image: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCABPg4UbkODVQ2eMrkVobSCh3HVn2lqoiyw&s",
      implemented: true
    },
    {
      name: "K-Nearest Neighbors",
      description: "Classify or regress based on closest training examples",
      path: "/knn",
      image: "https://cdn.iconscout.com/icon/premium/png-256-thumb/knn-algorithm-2130387-1794931.png",
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
      name: "Random Forest",
      description: "Ensemble method using multiple decision trees",
      path: "/random-forest",
      image: "https://cdn-icons-png.flaticon.com/512/6461/6461365.png",
      implemented: false
    },
    {
      name: "Support Vector Machine",
      description: "Classify data by finding optimal hyperplanes",
      path: "/svm",
      image: "https://cdn5.vectorstock.com/i/1000x1000/76/89/support-machine-svm-algorithm-color-icon-vector-50297689.jpg",
      implemented: true
    },
    {
      name: "Artificial Neural Network",
      description: "Deep learning model inspired by the human brain",
      path: "/ann",
      image: "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
      implemented: false
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
      image: "https://cdn-icons-png.flaticon.com/512/9464/9464616.png",
      implemented: true
    },
    {
      name: "Principal Component Analysis",
      description: "Reduce dimensionality while preserving variance",
      path: "/pca",
      image: "https://cdn-icons-png.flaticon.com/512/4616/4616734.png",
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

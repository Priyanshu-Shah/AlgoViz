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

      {/* Models Grid */}
      <motion.div 
        className="model-grid"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Linear Regression Card */}
        <motion.div
          className="model-card"
          onClick={() => navigate('/lin-reg')}
          variants={itemVariants}
          whileHover={{ 
            scale: 1.05, 
            boxShadow: "0px 10px 20px rgba(0,0,0,0.15)" 
          }}
          whileTap={{ scale: 0.98 }}
        >
          <div className="card-content">
            <div className="card-image-container">
              <motion.img
                src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCABPg4UbkODVQ2eMrkVobSCh3HVn2lqoiyw&s"
                alt="Linear Regression"
                className="model-image"
                whileHover={{ rotate: 5 }}
                transition={{ type: "spring", stiffness: 200 }}
              />
            </div>
            <h3 className="model-name">Linear Regression</h3>
            <p className="model-description">Predict continuous values based on linear relationships</p>
            <div className="card-cta">
              <span>Explore</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <path d="M12 5L19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
          </div>
        </motion.div>

        {/* Decision Tree Card */}
        <motion.div
          className="model-card"
          onClick={() => navigate('/dec-tree')}
          variants={itemVariants}
          whileHover={{ 
            scale: 1.05, 
            boxShadow: "0px 10px 20px rgba(0,0,0,0.15)" 
          }}
          whileTap={{ scale: 0.98 }}
        >
          <div className="card-content">
            <div className="card-image-container">
              <motion.img
                src="https://static.thenounproject.com/png/1119933-200.png"
                alt="Decision Tree"
                className="model-image"
                whileHover={{ rotate: 5 }}
                transition={{ type: "spring", stiffness: 200 }}
              />
            </div>
            <h3 className="model-name">Decision Tree</h3>
            <p className="model-description">Make decisions based on hierarchical feature splits</p>
            <div className="card-cta">
              <span>Explore</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <path d="M12 5L19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
          </div>
        </motion.div>
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

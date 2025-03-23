import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="homepage">
      <h1 className="homepage-title">Machine Learning Models</h1>
      <div className="model-grid">
        {/* Linear Regression Section */}
        <div
          className="model-card"
          onClick={() => navigate('/lin-reg')}
        >
          <img
            src="data:image/png;base64,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"
            alt="Linear Regression"
            className="model-image"
          />
          <p className="model-name">Linear Regression</p>
        </div>

        {/* Decision Tree Section */}
        <div
          className="model-card"
          onClick={() => navigate('/decision-tree')}
        >
          <img
            src="https://static.thenounproject.com/png/1119933-200.png"
            alt="Decision Tree"
            className="model-image"
          />
          <p className="model-name">Decision Tree</p>
        </div>

        {/* Add more models manually here */}
      </div>
    </div>
  );
}

export default HomePage;
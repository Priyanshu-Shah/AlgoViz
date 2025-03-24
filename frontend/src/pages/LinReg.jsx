import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { runLinearRegression, getLinearRegressionSampleData, checkHealth } from '../api';
import './ModelPage.css';

function LinReg() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [sampleLoading, setSampleLoading] = useState(false);

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        // console.log("Checking backend health at: " + Process.env.API_URL + "/health");
        const health = await checkHealth();
        console.log("Backend health response:", health);
        setBackendStatus(health.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        console.error("Backend health check failed:", err);
        setBackendStatus("disconnected");
        setError("Backend connection error: " + (err.message || "Unknown error"));
      }
    };
    
    checkBackendHealth();
  }, []);

  const handleAddPair = () => {
    setDataPairs([...dataPairs, { x: '', y: '' }]);
  };

  const handleRemovePair = (index) => {
    const newPairs = [...dataPairs];
    newPairs.splice(index, 1);
    setDataPairs(newPairs);
  };

  const handleInputChange = (index, field, value) => {
    const newPairs = [...dataPairs];
    
    // Strip any non-numeric characters except decimal point and minus sign
    if (field === 'x' || field === 'y') {
      // Allow only numbers, decimal point, and minus sign at the beginning
      value = value.replace(/[^0-9.-]/g, '');
      
      // Ensure only one decimal point
      const parts = value.split('.');
      if (parts.length > 2) {
        value = parts[0] + '.' + parts.slice(1).join('');
      }
      
      // Ensure minus sign is only at the beginning
      if (value.indexOf('-') > 0) {
        value = value.replace(/-/g, '');
        value = '-' + value;
      }
    }
    
    newPairs[index][field] = value;
    setDataPairs(newPairs);
  };

  const loadSampleData = async () => {
    setSampleLoading(true);
    setError(null);
    
    try {
      const sampleData = await getLinearRegressionSampleData();
      
      // Convert sample data into pairs
      const pairs = sampleData.X.map((x, index) => ({
        x: x.toString(),
        y: sampleData.y[index].toString()
      }));
      
      setDataPairs(pairs);
    } catch (err) {
      setError('Failed to load sample data. Please try again.');
      console.error(err);
    } finally {
      setSampleLoading(false);
    }
  };

  const handleRunModel = async () => {
    // Validate inputs
    if (dataPairs.length < 3) {
      setError('Please add at least 3 data points for meaningful regression');
      return;
    }

    // More detailed validation to identify the problem point
    const invalidPairs = dataPairs.filter(pair => 
      pair.x === '' || pair.y === '' || isNaN(parseFloat(pair.x)) || isNaN(parseFloat(pair.y))
    );
    
    if (invalidPairs.length > 0) {
      console.log('Invalid pairs found:', invalidPairs);
      
      // Create a more specific error message
      const errorMessage = `Invalid input found at row(s): ${
        dataPairs.map((pair, idx) => 
          (pair.x === '' || pair.y === '' || isNaN(parseFloat(pair.x)) || isNaN(parseFloat(pair.y))) ? (idx + 1) : null
        ).filter(Boolean).join(', ')
      }. All inputs must be valid numbers.`;
      
      setError(errorMessage);
      return;
    }

    setError(null);
    setLoading(true);
    setResults(null); // Reset results before new request

    try {
      // Log the data before sending to API for debugging
      console.log('Preparing data for API:', {
        X: dataPairs.map(pair => parseFloat(pair.x.trim())),
        y: dataPairs.map(pair => parseFloat(pair.y.trim()))
      });
      
      // Prepare data for the API with extra validation
      const apiData = {
        X: dataPairs.map(pair => {
          const x = parseFloat(pair.x.trim());
          if (isNaN(x)) throw new Error(`Invalid X value: ${pair.x}`);
          return x;
        }),
        y: dataPairs.map(pair => {
          const y = parseFloat(pair.y.trim());
          if (isNaN(y)) throw new Error(`Invalid Y value: ${pair.y}`);
          return y;
        })
      };

      console.log('Calling API endpoint...');
      const response = await runLinearRegression(apiData);


      
      if (response.error) {
        throw new Error(response.error);
      }
      
      setResults(response);
      console.log('Results state set to:', response);

      

    } catch (err) {
      console.error('Error details:', err);
      setError(`Error: ${err.message || 'An error occurred while running the model. Please try again.'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div 
      className="model-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
        </button>
        <h1 className="model-title">Linear Regression</h1>
      </div>

      <p className="model-description">
        Linear regression is a supervised learning algorithm used to predict a continuous target variable based on one or more predictor variables. 
        It assumes a linear relationship between the input variables and the target.
      </p>

      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>
            Backend service is not responding. Please make sure the Flask server is running on port 5000.
          </span>
        </div>
      )}

      <div className="content-container" style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '2rem',
        alignItems: 'start'
      }}>
        <div className="input-section">
          <div className="section-header">
            <h2 className="section-title">Input Data</h2>
            <button 
              className="sample-data-button" 
              onClick={loadSampleData}
              disabled={sampleLoading}
            >
              {sampleLoading ? 'Loading...' : 'Load Sample Data'}
            </button>
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <div>
            <p style={{ marginBottom: '1rem', color: '#4b5563' }}>
              Enter (x, y) pairs for regression analysis:
            </p>
            
            {dataPairs.map((pair, index) => (
              <div key={index} className="data-input-container">
                <div className="data-pair-input">
                  <input
                    type="text"
                    className="input-control"
                    placeholder="X value (e.g., 1.5)"
                    value={pair.x}
                    onChange={(e) => handleInputChange(index, 'x', e.target.value)}
                  />
                  <input
                    type="text"
                    className="input-control"
                    placeholder="Y value (e.g., 7.2)"
                    value={pair.y}
                    onChange={(e) => handleInputChange(index, 'y', e.target.value)}
                  />
                </div>
                {dataPairs.length > 1 && (
                  <button 
                    className="remove-btn" 
                    onClick={() => handleRemovePair(index)}
                    aria-label="Remove data point"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            
            <button 
              className="add-data-button" 
              onClick={handleAddPair}
            >
              + Add Data Point
            </button>
          </div>
          
          <div style={{ marginTop: '2rem' }}>
            <button 
              className="action-button" 
              onClick={handleRunModel}
              disabled={loading || backendStatus === "disconnected"}
            >
              {loading ? 'Running...' : 'Run Regression'}
            </button>
          </div>
        </div>

        <div className="results-section">
          <h2 className="section-title">Results</h2>
          {loading ? (
            <div className="loading-spinner">
              <div className="spinner"></div>
            </div>
          ) : results ? (
            <div className="results-content" style={{ padding: '20px', background: '#f9fafb', borderRadius: '8px' }}>
              <div className="metric-card">
                <div className="metric-title">Model Equation</div>
                <div className="metric-value">
                  {`y = ${results.coefficient !== undefined ? results.coefficient.toFixed(4) : '?'} × x + ${results.intercept !== undefined ? results.intercept.toFixed(4) : '?'}`}
                </div>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                <div className="metric-card">
                  <div className="metric-title">Coefficient</div>
                  <div className="metric-value">
                    {results.coefficient !== undefined ? results.coefficient.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-title">Intercept</div>
                  <div className="metric-value">
                    {results.intercept !== undefined ? results.intercept.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-title">R² Score</div>
                  <div className="metric-value">
                    {results.r2 !== undefined ? results.r2.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-title">Mean Squared Error</div>
                  <div className="metric-value">
                    {results.mse !== undefined ? results.mse.toFixed(4) : 'N/A'}
                  </div>
                </div>
              </div>
              
              <div className="visualization-container">
                <h3>Visualization:</h3>
                {results.plot ? (
                  <img 
                    src={`data:image/png;base64,${results.plot}`} 
                    alt="Linear Regression Plot" 
                    style={{ maxWidth: '100%', border: '1px solid #e5e7eb' }}
                  />
                ) : (
                  <p>No plot data found in response</p>
                )}
              </div>
            </div>
          ) : (
            <div style={{ color: '#6b7280', textAlign: 'center', padding: '2rem' }}>
              Enter data points and run the model to see results.
            </div>
          )}
        </div>
      </div>
  
    </motion.div>
  );
}

export default LinReg;
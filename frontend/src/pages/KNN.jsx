import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { runKnnClassification, runKnnRegression } from '../api';
import './ModelPage.css';

function KNN() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('classification');
  const [dataPairs, setDataPairs] = useState([{ x1: '', x2: '', y: '' }]);
  const [neighbors, setNeighbors] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleAddPair = () => {
    if (activeTab === 'classification') {
      setDataPairs([...dataPairs, { x1: '', x2: '', y: '' }]);
    } else {
      setDataPairs([...dataPairs, { x: '', y: '' }]);
    }
  };

  const handleRemovePair = (index) => {
    const newPairs = [...dataPairs];
    newPairs.splice(index, 1);
    setDataPairs(newPairs);
  };

  const handleInputChange = (index, field, value) => {
    const newPairs = [...dataPairs];
    newPairs[index][field] = value;
    setDataPairs(newPairs);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResults(null);
    setError(null);
    
    // Reset data pairs based on the tab
    if (tab === 'classification') {
      setDataPairs([{ x1: '', x2: '', y: '' }]);
    } else {
      setDataPairs([{ x: '', y: '' }]);
    }
  };

  const handleRunModel = async () => {
    // Validate inputs
    if (dataPairs.length < 5) {
      setError('Please add at least 5 data points for meaningful KNN analysis');
      return;
    }

    // Check for empty or invalid inputs
    let hasInvalidInputs = false;
    
    if (activeTab === 'classification') {
      hasInvalidInputs = dataPairs.some(pair => 
        pair.x1 === '' || pair.x2 === '' || pair.y === '' || 
        isNaN(parseFloat(pair.x1)) || isNaN(parseFloat(pair.x2))
      );
    } else {
      hasInvalidInputs = dataPairs.some(pair => 
        pair.x === '' || pair.y === '' || isNaN(parseFloat(pair.x)) || isNaN(parseFloat(pair.y))
      );
    }
    
    if (hasInvalidInputs) {
      setError('All inputs must be valid numbers (except class labels for classification)');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      // Prepare data for the API
      let apiData;
      
      if (activeTab === 'classification') {
        apiData = {
          X: dataPairs.map(pair => [parseFloat(pair.x1), parseFloat(pair.x2)]),
          y: dataPairs.map(pair => pair.y),
          n_neighbors: parseInt(neighbors)
        };
        
        const response = await runKnnClassification(apiData);
        setResults(response);
      } else {
        apiData = {
          X: dataPairs.map(pair => [parseFloat(pair.x)]),
          y: dataPairs.map(pair => parseFloat(pair.y)),
          n_neighbors: parseInt(neighbors)
        };
        
        const response = await runKnnRegression(apiData);
        setResults(response);
      }
    } catch (err) {
      setError('An error occurred while running the model. Please try again.');
      console.error(err);
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
        <h1 className="model-title">K-Nearest Neighbors (KNN)</h1>
      </div>

      <p className="model-description">
        K-Nearest Neighbors is a versatile machine learning algorithm used for both classification and regression tasks. 
        It makes predictions based on the k most similar data points in the training set.
      </p>

      <div className="tabs">
        <div 
          className={`tab ${activeTab === 'classification' ? 'active' : ''}`}
          onClick={() => handleTabChange('classification')}
        >
          Classification
        </div>
        <div 
          className={`tab ${activeTab === 'regression' ? 'active' : ''}`}
          onClick={() => handleTabChange('regression')}
        >
          Regression
        </div>
      </div>

      <div className="content-container">
        <div className="input-section">
          <h2 className="section-title">Input Data</h2>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className="form-group">
            <label htmlFor="neighbors">Number of Neighbors (k):</label>
            <input
              id="neighbors"
              type="number"
              className="input-control"
              min="1"
              value={neighbors}
              onChange={(e) => setNeighbors(e.target.value)}
            />
          </div>
          
          <div>
            {activeTab === 'classification' ? (
              <>
                <p style={{ marginBottom: '1rem', color: '#4b5563' }}>
                  Enter (x1, x2, class) values for classification:
                </p>
                
                {dataPairs.map((pair, index) => (
                  <div key={index} className="data-input-container">
                    <div className="data-pair-input">
                      <input
                        type="text"
                        className="input-control"
                        placeholder="X1 value"
                        value={pair.x1}
                        onChange={(e) => handleInputChange(index, 'x1', e.target.value)}
                      />
                      <input
                        type="text"
                        className="input-control"
                        placeholder="X2 value"
                        value={pair.x2}
                        onChange={(e) => handleInputChange(index, 'x2', e.target.value)}
                      />
                      <input
                        type="text"
                        className="input-control"
                        placeholder="Class"
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
              </>
            ) : (
              <>
                <p style={{ marginBottom: '1rem', color: '#4b5563' }}>
                  Enter (x, y) pairs for regression:
                </p>
                
                {dataPairs.map((pair, index) => (
                  <div key={index} className="data-input-container">
                    <div className="data-pair-input">
                      <input
                        type="text"
                        className="input-control"
                        placeholder="X value"
                        value={pair.x}
                        onChange={(e) => handleInputChange(index, 'x', e.target.value)}
                      />
                      <input
                        type="text"
                        className="input-control"
                        placeholder="Y value"
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
              </>
            )}
            
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
              disabled={loading}
            >
              {loading ? 'Running...' : 'Run KNN'}
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
            <div className="results-content">
              {activeTab === 'classification' ? (
                <>
                  <div className="metric-card">
                    <div className="metric-title">Accuracy</div>
                    <div className="metric-value">{(results.accuracy * 100).toFixed(2)}%</div>
                  </div>
                  
                  <div className="metric-card">
                    <div className="metric-title">Model Parameters</div>
                    <div className="metric-value">k = {results.n_neighbors}</div>
                  </div>
                  
                  <div className="visualization-container">
                    <h3 style={{ marginBottom: '1rem' }}>Classification Visualization</h3>
                    {results.plot && (
                      <img 
                        src={`data:image/png;base64,${results.plot}`} 
                        alt="KNN Classification Plot" 
                      />
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="metric-card">
                      <div className="metric-title">R² Score (Train)</div>
                      <div className="metric-value">{results.r2_train.toFixed(4)}</div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-title">R² Score (Test)</div>
                      <div className="metric-value">{results.r2_test.toFixed(4)}</div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-title">MSE (Train)</div>
                      <div className="metric-value">{results.mse_train.toFixed(4)}</div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-title">MSE (Test)</div>
                      <div className="metric-value">{results.mse_test.toFixed(4)}</div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-title">Model Parameters</div>
                      <div className="metric-value">k = {results.n_neighbors}</div>
                    </div>
                  </div>
                  
                  <div className="visualization-container">
                    <h3 style={{ marginBottom: '1rem' }}>Regression Visualization</h3>
                    {results.plot && (
                      <img 
                        src={`data:image/png;base64,${results.plot}`} 
                        alt="KNN Regression Plot" 
                      />
                    )}
                  </div>
                </>
              )}
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

export default KNN;
 
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './ModelPage.css';

function KNN() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('classification');
  const [neighbors, setNeighbors] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // States for visualization
  const canvasRef = useRef(null);
  const [selectedClass, setSelectedClass] = useState('1');
  const [pointsMode, setPointsMode] = useState('train'); // 'train' or 'predict'
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 500, height: 500 });
  const [scale, setScale] = useState({
    x: { min: 0, max: 10 },
    y: { min: 0, max: 10 }
  });
  
  // For point input dialog
  const [showPointDialog, setShowPointDialog] = useState(false);
  const [dialogPosition, setDialogPosition] = useState({ x: 0, y: 0 });
  const [tempPoint, setTempPoint] = useState(null);
  const [tempValue, setTempValue] = useState('');
  
  // For decision boundary
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(false);
  const [boundaryImg, setBoundaryImg] = useState(null);
  
  // Handle tab change
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    resetVisualization();
  };

  // Convert screen coordinates to data coordinates
  const screenToData = (x, y) => {
    const { width, height } = canvasDimensions;
    const dataX = (x / width) * (scale.x.max - scale.x.min) + scale.x.min;
    const dataY = ((height - y) / height) * (scale.y.max - scale.y.min) + scale.y.min;
    return { x: dataX, y: dataY };
  };

  // Handle canvas click for classification
  const handleClassificationCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    if (pointsMode === 'train') {
      // Add point to training data - ENSURE CLASS IS A STRING
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y,
        y: selectedClass // Explicitly convert to string
      };
      
      console.log(`Adding training point with class: ${newPoint.y} (${typeof newPoint.y})`);
      
      setTrainingPoints([...trainingPoints, newPoint]);
      // Hide decision boundary if it was showing
      if (showDecisionBoundary) {
        setShowDecisionBoundary(false);
        setBoundaryImg(null);
      }
    } else {
      // Add point to predict list
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y
      };
      
      setPredictPoints([...predictPoints, newPoint]);
    }
  };
  
  // Handle canvas click for regression
  const handleRegressionCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    if (pointsMode === 'train') {
      // Store temporary point and show dialog
      setTempPoint({
        x1: dataPoint.x,
        x2: dataPoint.y
      });
      
      // Position dialog near click but ensure it stays on screen
      const dialogWidth = 200;
      const dialogHeight = 120;
      let dialogX = e.clientX - (dialogWidth / 2);
      let dialogY = e.clientY - dialogHeight - 10; // position above cursor
      
      // Adjust if off screen
      if (dialogX < 10) dialogX = 10;
      if (dialogX > window.innerWidth - dialogWidth - 10) dialogX = window.innerWidth - dialogWidth - 10;
      if (dialogY < 10) dialogY = e.clientY + 10; // position below cursor instead
      
      setDialogPosition({ x: dialogX, y: dialogY });
      setTempValue('');
      setShowPointDialog(true);
      
      // Hide decision boundary if it was showing
      if (showDecisionBoundary) {
        setShowDecisionBoundary(false);
        setBoundaryImg(null);
      }
    } else {
      // Add point to predict list
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y
      };
      
      setPredictPoints([...predictPoints, newPoint]);
    }
  };
  
  // Handle dialog confirmation
  const handleDialogConfirm = () => {
    // Validate input is a number
    const numValue = parseFloat(tempValue);
    
    if (tempPoint && !isNaN(numValue)) {
      // Add point to training data with value
      const newPoint = {
        x1: tempPoint.x1,
        x2: tempPoint.x2,
        y: tempValue
      };
      
      setTrainingPoints([...trainingPoints, newPoint]);
      setShowPointDialog(false);
      setTempPoint(null);
      setError(null); // Clear any errors
    } else {
      // Show error if not a valid number
      setError('Please enter a valid number for the value');
      // Keep dialog open
    }
  };
  
  // Handle canvas click based on active tab
  const handleCanvasClick = (e) => {
    if (activeTab === 'classification') {
      handleClassificationCanvasClick(e);
    } else {
      handleRegressionCanvasClick(e);
    }
  };

  // Make predictions for all points in predictPoints
  const predictAllPoints = async () => {
    if (trainingPoints.length < 1) {
      setError('Need at least 1 training point to make predictions');
      return;
    }
    
    if (predictPoints.length < 1) {
      setError('No prediction points added. Click in predict mode to add points to predict.');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const predictions = [];
      
      // Make prediction for each point
      for (const point of predictPoints) {
        // ENSURE y values are consistently strings in classification mode
        const apiData = {
          X: trainingPoints.map(p => [p.x1, p.x2]),
          y: trainingPoints.map(p => activeTab === 'classification' ? p.y.toString() : p.y),
          n_neighbors: parseInt(neighbors),
          predict_point: [point.x1, point.x2]
        };
        
        console.log("Sending prediction request with data:", {
          "X sample": apiData.X.slice(0, 2),
          "y sample": apiData.y.slice(0, 2),
          "y types": apiData.y.slice(0, 2).map(y => typeof y),
          "predict_point": apiData.predict_point,
          "n_neighbors": apiData.n_neighbors
        });
        
        const response = await fetch('http://localhost:5000/api/knn-predict-point', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(apiData),
        });
        
        const result = await response.json();
        
        if (result.error) {
          setError(result.error);
          setLoading(false);
          return;
        }
        
        console.log("Received prediction result:", result);
        
        predictions.push({
          ...point,
          predictedClass: result.predicted_class
        });
      }
      
      // Update predictions
      setPredictions(predictions);
      
    } catch (err) {
      setError('Error making prediction: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  // Generate decision boundary
  const generateDecisionBoundary = async () => {
    if (trainingPoints.length < 5) {
      setError('Need at least 5 training points to generate a decision boundary');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // ENSURE y values are consistently strings in classification mode
      const apiData = {
        X: trainingPoints.map(p => [p.x1, p.x2]),
        y: trainingPoints.map(p => activeTab === 'classification' ? p.y.toString() : p.y),
        n_neighbors: parseInt(neighbors)
      };
      
      console.log("Generating decision boundary with data:", {
        "X sample": apiData.X.slice(0, 2),
        "y sample": apiData.y.slice(0, 2),
        "y types": apiData.y.slice(0, 2).map(y => typeof y),
        "n_neighbors": apiData.n_neighbors
      });
      
      const response = await fetch('http://localhost:5000/api/knn-decision-boundary', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiData),
      });
      
      const result = await response.json();
      
      if (result.error) {
        setError(result.error);
        setLoading(false);
        return;
      }
      
      // Set the boundary image from base64 data
      if (result.decision_boundary) {
        setBoundaryImg(`data:image/png;base64,${result.decision_boundary}`);
        setShowDecisionBoundary(true);
      } else {
        setError('No decision boundary data received');
      }
      
    } catch (err) {
      setError('Error generating decision boundary: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Reset the visualization
  const resetVisualization = () => {
    setTrainingPoints([]);
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setError(null);
    setShowDecisionBoundary(false);
    setBoundaryImg(null);
  };

  // Handle Escape key press to close dialog
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && showPointDialog) {
        setShowPointDialog(false);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showPointDialog]);

  // Canvas drawing effect
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid and points
    drawCanvas();
    
    function drawCanvas() {
      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 0.5;
      
      // Draw horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = canvas.height - (i / 10) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }
      
      // Draw vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * canvas.width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      
      // Draw axes labels
      ctx.fillStyle = '#4b5563';
      ctx.font = '12px Inter, sans-serif';
      
      // X-axis labels
      for (let i = 0; i <= 10; i += 2) {
        const x = (i / 10) * canvas.width;
        const value = scale.x.min + (i / 10) * (scale.x.max - scale.x.min);
        ctx.fillText(value.toFixed(1), x - 10, canvas.height - 5);
      }
      
      // Y-axis labels
      for (let i = 0; i <= 10; i += 2) {
        const y = canvas.height - (i / 10) * canvas.height;
        const value = scale.y.min + (i / 10) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(1), 5, y + 4);
      }
      
      // Draw training points
      trainingPoints.forEach(point => {
        const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((point.x2 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        
        if (activeTab === 'classification') {
          // Different color based on class for classification - compare as string
          const pointClass = point.y.toString();
          if (pointClass === '1') {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue
          } else {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red
          }
        } else {
          // For regression, use a gradient based on value
          // Normalize value to be between 0 and 1 for color gradient
          const normalizedValue = Math.min(Math.max(parseFloat(point.y) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue);
          const g = Math.round(130 + (68 - 130) * normalizedValue);
          const b = Math.round(246 + (68 - 246) * normalizedValue);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
          
          // Add text label for the value
          ctx.fillStyle = '#000000';
          ctx.font = '10px Arial';
          ctx.fillText(point.y, x + 10, y - 10);
          
          // Reset fill style for the point
          const normalizedValue2 = Math.min(Math.max(parseFloat(point.y) / 10, 0), 1);
          const r2 = Math.round(59 + (239 - 59) * normalizedValue2);
          const g2 = Math.round(130 + (68 - 130) * normalizedValue2);
          const b2 = Math.round(246 + (68 - 246) * normalizedValue2);
          ctx.fillStyle = `rgba(${r2}, ${g2}, ${b2}, 0.7)`;
        }
        
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      
      // Draw points to predict (before prediction)
      if (predictions.length === 0) {
        predictPoints.forEach(point => {
          const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
          const y = canvas.height - ((point.x2 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
          
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          
          // Gray color for unpredicted points
          ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray
          
          ctx.fill();
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }
      
      // Draw prediction results
      predictions.forEach(pred => {
        const x = ((pred.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((pred.x2 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        
        if (activeTab === 'classification') {
          // Fill color based on predicted class with flexible matching
          const predClass = pred.predictedClass.toString();
          console.log(`Drawing prediction with class: ${predClass} (${typeof predClass})`);
          
          // If class starts with 1 or equals 1 or 1.0, etc. -> blue
          if (predClass.startsWith('1') || Math.round(parseFloat(predClass)) === 1) {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue
          } 
          // If class starts with 2 or equals 2 or 2.0, etc. -> red
          else if (predClass.startsWith('2') || Math.round(parseFloat(predClass)) === 2) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red
          } 
          // Fallback for any other values
          else {
            ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray
          }
        } else {
          // For regression, use a gradient based on predicted value
          const normalizedValue = Math.min(Math.max(parseFloat(pred.predictedClass) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue);
          const g = Math.round(130 + (68 - 130) * normalizedValue);
          const b = Math.round(246 + (68 - 246) * normalizedValue);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
          
          // Add text label for the predicted value
          ctx.fillStyle = '#000000';
          ctx.font = '10px Arial';
          ctx.fillText(pred.predictedClass, x + 10, y - 10);
          
          // Reset fill style for the point
          const normalizedValue2 = Math.min(Math.max(parseFloat(pred.predictedClass) / 10, 0), 1);
          const r2 = Math.round(59 + (239 - 59) * normalizedValue2);
          const g2 = Math.round(130 + (68 - 130) * normalizedValue2);
          const b2 = Math.round(246 + (68 - 246) * normalizedValue2);
          ctx.fillStyle = `rgba(${r2}, ${g2}, ${b2}, 0.7)`;
        }
        
        // Dotted border to indicate prediction
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }
  }, [trainingPoints, predictPoints, predictions, scale, canvasDimensions, activeTab]);

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
          <h2 className="section-title">KNN {activeTab === 'classification' ? 'Classification' : 'Regression'}</h2>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className="form-group">
            <label htmlFor="neighbors-slider" style={{ 
              display: 'block', 
              marginBottom: '0.5rem', 
              fontWeight: '500', 
              color: '#4b5563' 
            }}>
              Number of Neighbors (k): {neighbors}
            </label>
            <input
              id="neighbors-slider"
              type="range"
              min="1"
              max="10"
              step="1"
              value={neighbors}
              onChange={(e) => setNeighbors(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              fontSize: '0.8rem', 
              color: '#6b7280',
              marginTop: '0.25rem'
            }}>
              <span>1 (More flexible)</span>
              <span>10 (More stable)</span>
            </div>
          </div>
          
          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500' }}>
              Add Points Mode: {pointsMode === 'train' ? 'Training Points' : 'Points to Predict'}
            </h3>
            
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              <button
                onClick={() => setPointsMode('train')}
                style={{
                  padding: '0.75rem 1rem',
                  border: 'none',
                  borderRadius: '0.5rem',
                  backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb',
                  color: pointsMode === 'train' ? 'white' : '#4b5563',
                  cursor: 'pointer',
                  fontWeight: '500',
                  boxShadow: pointsMode === 'train' ? '0 2px 4px rgba(59, 130, 246, 0.3)' : 'none',
                  transition: 'all 0.2s ease'
                }}
              >
                Add Training Points
              </button>
              <button
                onClick={() => setPointsMode('predict')}
                style={{
                  padding: '0.75rem 1rem',
                  border: 'none',
                  borderRadius: '0.5rem',
                  backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb',
                  color: pointsMode === 'predict' ? 'white' : '#4b5563',
                  cursor: 'pointer',
                  fontWeight: '500',
                  boxShadow: pointsMode === 'predict' ? '0 2px 4px rgba(59, 130, 246, 0.3)' : 'none',
                  transition: 'all 0.2s ease'
                }}
              >
                Add Prediction Points
              </button>
            </div>
            
            {pointsMode === 'train' && activeTab === 'classification' && (
              <div style={{ marginBottom: '1.5rem' }}>
                <p style={{ marginBottom: '0.5rem', color: '#4b5563' }}>
                  Select class for training points:
                </p>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={() => setSelectedClass('1')}
                    style={{
                      padding: '0.75rem 1.25rem',
                      border: 'none',
                      borderRadius: '0.5rem',
                      backgroundColor: selectedClass === '1' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)',
                      color: selectedClass === '1' ? 'white' : '#1e40af',
                      cursor: 'pointer',
                      fontWeight: '500',
                      boxShadow: selectedClass === '1' ? '0 2px 4px rgba(59, 130, 246, 0.3)' : 'none',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Class 1
                  </button>
                  <button
                    onClick={() => setSelectedClass('2')}
                    style={{
                      padding: '0.75rem 1.25rem',
                      border: 'none',
                      borderRadius: '0.5rem',
                      backgroundColor: selectedClass === '2' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)',
                      color: selectedClass === '2' ? 'white' : '#b91c1c',
                      cursor: 'pointer',
                      fontWeight: '500',
                      boxShadow: selectedClass === '2' ? '0 2px 4px rgba(239, 68, 68, 0.3)' : 'none',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Class 2
                  </button>
                </div>
              </div>
            )}
            
            {pointsMode === 'train' && activeTab === 'regression' && (
              <div style={{ marginBottom: '1.5rem' }}>
                <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                  Click on the graph to add training points. You'll be prompted to enter a value for each point.
                </p>
              </div>
            )}
            
            <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <button 
                className="action-button" 
                onClick={predictAllPoints}
                disabled={loading || trainingPoints.length < 1 || predictPoints.length < 1}
                style={{
                  backgroundColor: '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '1rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 2px 4px rgba(59, 130, 246, 0.3)',
                  opacity: (loading || trainingPoints.length < 1 || predictPoints.length < 1) ? 0.7 : 1
                }}
              >
                {loading ? 'Predicting...' : 'Predict Points'}
              </button>
              
              <button 
                className="action-button" 
                onClick={generateDecisionBoundary}
                disabled={loading || trainingPoints.length < 5}
                style={{
                  backgroundColor: '#8b5cf6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '1rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 2px 4px rgba(139, 92, 246, 0.3)',
                  opacity: (loading || trainingPoints.length < 5) ? 0.7 : 1
                }}
              >
                {loading ? 'Generating...' : 'Show Decision Regions'}
              </button>
              
              <button 
                className="action-button" 
                onClick={resetVisualization}
                style={{
                  backgroundColor: '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '1rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 2px 4px rgba(239, 68, 68, 0.3)',
                }}
              >
                Reset All
              </button>
            </div>
            
            <div style={{ marginTop: '1.5rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem' }}>
              <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>
                Statistics:
              </p>
              <p>
                <strong>Training Points:</strong> {trainingPoints.length}
              </p>
              <p>
                <strong>Points to Predict:</strong> {predictPoints.length}
              </p>
              <p>
                <strong>Predictions Made:</strong> {predictions.length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="results-section">
          <h2 className="section-title">Visualization</h2>
          
          {loading ? (
            <div className="loading-spinner">
              <div className="spinner"></div>
            </div>
          ) : (
            <div style={{ 
              border: '1px solid #e5e7eb', 
              borderRadius: '0.75rem', 
              overflow: 'hidden',
              position: 'relative',
              backgroundColor: '#f9fafb',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
            }}>
              {showDecisionBoundary && boundaryImg && (
                <img 
                  src={boundaryImg}
                  alt="Decision Boundary"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    zIndex: 1,
                    objectFit: 'cover',
                    pointerEvents: 'none',
                  }}
                />
              )}
              
              <canvas 
                ref={canvasRef}
                width={canvasDimensions.width}
                height={canvasDimensions.height}
                onClick={handleCanvasClick}
                style={{ 
                  display: 'block', 
                  cursor: 'crosshair',
                  position: 'relative',
                  zIndex: 2
                }}
              />
              
              <div style={{
                position: 'absolute',
                bottom: '10px',
                right: '10px',
                padding: '0.5rem 0.75rem',
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderRadius: '0.375rem',
                fontSize: '0.875rem',
                fontWeight: '500',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
                zIndex: 3
              }}>
                <strong>Mode: {pointsMode === 'train' ? 'Adding Training Points' : 'Adding Points to Predict'}</strong>
              </div>
            </div>
          )}
          
          <div style={{ marginTop: '1rem', backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '0.5rem' }}>
            <h3 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500' }}>Legend:</h3>
            {activeTab === 'classification' ? (
              <div>
                <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.375rem' }}>
                  <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: 'rgba(59, 130, 246, 0.7)', borderRadius: '50%', marginRight: '8px' }}></span>
                  <strong>Blue points:</strong> Class 1
                </p>
                <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.375rem' }}>
                  <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: 'rgba(239, 68, 68, 0.7)', borderRadius: '50%', marginRight: '8px' }}></span>
                  <strong>Red points:</strong> Class 2
                </p>
              </div>
            ) : (
              <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.375rem' }}>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: 'linear-gradient(to right, rgba(59, 130, 246, 0.7), rgba(239, 68, 68, 0.7))', borderRadius: '50%', marginRight: '8px' }}></span>
                <strong>Point colors:</strong> Color gradient represents values (blue → red = low → high)
              </p>
            )}
            <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.375rem' }}>
              <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: 'rgba(156, 163, 175, 0.7)', borderRadius: '50%', marginRight: '8px' }}></span>
              <strong>Gray points:</strong> Unpredicted points
            </p>
            <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>
              <span style={{ display: 'inline-block', width: '12px', height: '12px', border: '1px solid black', borderRadius: '50%', marginRight: '8px' }}></span>
              <strong>Solid border:</strong> Training point | 
              <span style={{ display: 'inline-block', width: '12px', height: '12px', border: '1px dashed black', borderRadius: '50%', marginRight: '8px', marginLeft: '8px' }}></span>
              <strong>Dotted border:</strong> Predicted point
            </p>
          </div>
        </div>
      </div>
      
      {/* Point Value Dialog */}
      {showPointDialog && (
        <div 
          style={{
            position: 'fixed',
            top: dialogPosition.y,
            left: dialogPosition.x,
            backgroundColor: 'white',
            padding: '1rem',
            borderRadius: '0.5rem',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            zIndex: 100,
            width: '200px'
          }}
        >
          <p style={{ marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: '500' }}>
            Enter value for this point:
          </p>
          <input
            type="text"
            value={tempValue}
            onChange={(e) => setTempValue(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #e5e7eb',
              borderRadius: '0.375rem',
              marginBottom: '0.75rem'
            }}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleDialogConfirm();
              } else if (e.key === 'Escape') {
                setShowPointDialog(false);
              }
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button
              onClick={() => setShowPointDialog(false)}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#f3f4f6',
                color: '#4b5563',
                border: 'none',
                borderRadius: '0.375rem',
                fontSize: '0.875rem',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleDialogConfirm}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '0.375rem',
                fontSize: '0.875rem',
                cursor: 'pointer'
              }}
            >
              Add
            </button>
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default KNN;
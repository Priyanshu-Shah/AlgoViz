import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './ModelPage.css';

function KNN() {
  const navigate = useNavigate();
  const [neighbors, setNeighbors] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // States for interactive visualization
  const canvasRef = useRef(null);
  const [selectedClass, setSelectedClass] = useState('1');
  const [pointsMode, setPointsMode] = useState('train'); // 'train' or 'predict'
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 400, height: 400 });
  const [scale, setScale] = useState({
    x: { min: 0, max: 10 },
    y: { min: 0, max: 10 }
  });

  // Convert screen coordinates to data coordinates
  const screenToData = (x, y) => {
    const { width, height } = canvasDimensions;
    const dataX = (x / width) * (scale.x.max - scale.x.min) + scale.x.min;
    const dataY = ((height - y) / height) * (scale.y.max - scale.y.min) + scale.y.min;
    return { x: dataX, y: dataY };
  };

  // Handle canvas click
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    if (pointsMode === 'train') {
      // Add point to training data
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y,
        y: selectedClass
      };
      
      setTrainingPoints([...trainingPoints, newPoint]);
    } else {
      // Add point to predict list
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y
      };
      
      setPredictPoints([...predictPoints, newPoint]);
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
        const apiData = {
          X: trainingPoints.map(p => [p.x1, p.x2]),
          y: trainingPoints.map(p => p.y),
          n_neighbors: parseInt(neighbors),
          predict_point: [point.x1, point.x2]
        };
        
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
          return;
        }
        
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

  // Reset the visualization
  const resetVisualization = () => {
    setTrainingPoints([]);
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setError(null);
  };

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
        
        // Different color based on class
        if (point.y === '1') {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue
        } else {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red
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
        
        // Fill color based on predicted class
        if (pred.predictedClass === '1') {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue
        } else {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red
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
  }, [trainingPoints, predictPoints, predictions, scale, canvasDimensions]);

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
        K-Nearest Neighbors is a versatile machine learning algorithm used for classification tasks.
        It makes predictions based on the k most similar data points in the training set.
      </p>

      <div className="content-container">
        <div className="input-section">
          <h2 className="section-title">Interactive KNN</h2>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className="form-group">
            <label htmlFor="interactive-neighbors">Number of Neighbors (k):</label>
            <input
              id="interactive-neighbors"
              type="number"
              className="input-control"
              min="1"
              value={neighbors}
              onChange={(e) => setNeighbors(e.target.value)}
            />
          </div>
          
          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500' }}>
              Add Points Mode: {pointsMode === 'train' ? 'Training Points' : 'Points to Predict'}
            </h3>
            
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              <button
                onClick={() => setPointsMode('train')}
                style={{
                  padding: '0.5rem 1rem',
                  border: 'none',
                  borderRadius: '0.375rem',
                  backgroundColor: pointsMode === 'train' ? '#4b5563' : '#e5e7eb',
                  color: pointsMode === 'train' ? 'white' : '#4b5563',
                  cursor: 'pointer'
                }}
              >
                Add Training Points
              </button>
              <button
                onClick={() => setPointsMode('predict')}
                style={{
                  padding: '0.5rem 1rem',
                  border: 'none',
                  borderRadius: '0.375rem',
                  backgroundColor: pointsMode === 'predict' ? '#4b5563' : '#e5e7eb',
                  color: pointsMode === 'predict' ? 'white' : '#4b5563',
                  cursor: 'pointer'
                }}
              >
                Add Prediction Points
              </button>
            </div>
            
            {pointsMode === 'train' && (
              <div style={{ marginBottom: '1rem' }}>
                <p style={{ marginBottom: '0.5rem', color: '#4b5563' }}>
                  Select class for training points:
                </p>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={() => setSelectedClass('1')}
                    style={{
                      padding: '0.5rem 1rem',
                      border: 'none',
                      borderRadius: '0.375rem',
                      backgroundColor: selectedClass === '1' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.2)',
                      color: selectedClass === '1' ? 'white' : '#1e40af',
                      cursor: 'pointer'
                    }}
                  >
                    Class 1
                  </button>
                  <button
                    onClick={() => setSelectedClass('2')}
                    style={{
                      padding: '0.5rem 1rem',
                      border: 'none',
                      borderRadius: '0.375rem',
                      backgroundColor: selectedClass === '2' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.2)',
                      color: selectedClass === '2' ? 'white' : '#b91c1c',
                      cursor: 'pointer'
                    }}
                  >
                    Class 2
                  </button>
                </div>
              </div>
            )}
            
            <p style={{ marginBottom: '1rem', color: '#4b5563' }}>
              {pointsMode === 'train' 
                ? 'Click on the graph to add training points' 
                : 'Click on the graph to add points you want to predict'}
            </p>
            
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <button 
                className="action-button" 
                onClick={predictAllPoints}
                disabled={loading || trainingPoints.length < 1 || predictPoints.length < 1}
              >
                {loading ? 'Predicting...' : 'Predict Points'}
              </button>
              
              <button 
                className="action-button" 
                onClick={resetVisualization}
                style={{ backgroundColor: '#ef4444' }}
              >
                Reset All
              </button>
            </div>
            
            <div style={{ marginTop: '1rem' }}>
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
          <h2 className="section-title">Interactive Visualization</h2>
          
          {loading ? (
            <div className="loading-spinner">
              <div className="spinner"></div>
            </div>
          ) : (
            <div style={{ 
              border: '1px solid #e5e7eb', 
              borderRadius: '0.5rem', 
              overflow: 'hidden',
              position: 'relative' 
            }}>
              <canvas 
                ref={canvasRef}
                width={canvasDimensions.width}
                height={canvasDimensions.height}
                onClick={handleCanvasClick}
                style={{ 
                  display: 'block', 
                  backgroundColor: '#f9fafb',
                  cursor: 'crosshair'
                }}
              />
              
              <div style={{
                position: 'absolute',
                bottom: '10px',
                right: '10px',
                padding: '0.5rem',
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderRadius: '0.25rem',
                fontSize: '0.875rem'
              }}>
                <strong>Mode: {pointsMode === 'train' ? 'Adding Training Points' : 'Adding Points to Predict'}</strong>
              </div>
            </div>
          )}
          
          <div style={{ marginTop: '1rem' }}>
            <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
              <strong>Blue points:</strong> Class 1 | <strong>Red points:</strong> Class 2 | <strong>Gray points:</strong> Unpredicted
            </p>
            <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
              Solid border = training point | Dotted border = predicted point
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default KNN;
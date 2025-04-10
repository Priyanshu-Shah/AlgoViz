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
  const [backendStatus, setBackendStatus] = useState("connected");
  
  // States for visualization
  const canvasRef = useRef(null);
  const boundaryCanvasRef = useRef(null);
  const [selectedClass, setSelectedClass] = useState('1');
  const [pointsMode, setPointsMode] = useState('train'); // 'train' or 'predict'
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const canvasDimensions = { width: 600, height: 600 }; // Square canvas for proper aspect ratio
  // Expanded scale from -8 to 8
  const scale = {
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  };
  
  // For point input dialog
  const [showPointDialog, setShowPointDialog] = useState(false);
  const [dialogPosition, setDialogPosition] = useState({ x: 0, y: 0 });
  const [tempPoint, setTempPoint] = useState(null);
  const [tempValue, setTempValue] = useState('');
  
  // For decision boundary
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(false);
  const [boundaryImg, setBoundaryImg] = useState(null);
  
  // For hover effect on nearest neighbors
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [nearestNeighbors, setNearestNeighbors] = useState([]);
  
  // Handle tab change
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    resetData();
  };

  // Improved coordinate transformation function
  const screenToData = (x, y) => {
    // Get the actual rendered size of the canvas (which might be different from the internal size)
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const scaleFactorX = canvas.width / rect.width;
    const scaleFactorY = canvas.height / rect.height;
    
    // Adjust the x and y by the scale factor
    const adjustedX = x * scaleFactorX;
    const adjustedY = y * scaleFactorY;
    
    // Now convert to data coordinates
    const dataX = scale.x.min + (adjustedX / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - (adjustedY / canvas.height) * (scale.y.max - scale.y.min);
    
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
      // Add point to training data
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y,
        y: selectedClass
      };
      
      console.log(`Adding training point at (${dataPoint.x.toFixed(2)}, ${dataPoint.y.toFixed(2)}) with class: ${newPoint.y}`);
      
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
  
  // Handle canvas mouse move for hover effect
  const handleCanvasMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    // Find if mouse is near any prediction point (within a small radius)
    const hoverRadius = 0.5; // Data units
    let foundPoint = null;
    
    // Check all prediction points
    for (const point of predictions.length > 0 ? predictions : predictPoints) {
      const dist = Math.sqrt(
        Math.pow(point.x1 - dataPoint.x, 2) + 
        Math.pow(point.x2 - dataPoint.y, 2)
      );
      
      if (dist < hoverRadius) {
        foundPoint = point;
        break;
      }
    }
    
    if (foundPoint !== hoveredPoint) {
      setHoveredPoint(foundPoint);
      
      if (foundPoint && trainingPoints.length > 0) {
        // Find k nearest neighbors to this point
        const k = neighbors;
        const distances = trainingPoints.map(tp => ({
          point: tp,
          distance: Math.sqrt(
            Math.pow(tp.x1 - foundPoint.x1, 2) + 
            Math.pow(tp.x2 - foundPoint.x2, 2)
          )
        }));
        
        // Sort by distance and take the k nearest
        distances.sort((a, b) => a.distance - b.distance);
        const nearest = distances.slice(0, k).map(d => d.point);
        
        setNearestNeighbors(nearest);
      } else {
        setNearestNeighbors([]);
      }
    }
  };
  
  // Handle canvas mouse leave
  const handleCanvasMouseLeave = () => {
    setHoveredPoint(null);
    setNearestNeighbors([]);
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
        // Send explicit mode to the backend
        const apiData = {
          X: trainingPoints.map(p => [p.x1, p.x2]),
          y: trainingPoints.map(p => p.y),
          n_neighbors: parseInt(neighbors),
          predict_point: [point.x1, point.x2],
          mode: activeTab // Explicitly send 'classification' or 'regression'
        };
        
        console.log("Sending prediction request with data:", {
          "X sample": apiData.X.slice(0, 2),
          "y sample": apiData.y.slice(0, 2),
          "y types": apiData.y.slice(0, 2).map(y => typeof y),
          "mode": apiData.mode,
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
      
      // Send explicit mode to the backend
      const apiData = {
        X: trainingPoints.map(p => [p.x1, p.x2]),
        y: trainingPoints.map(p => p.y),
        n_neighbors: parseInt(neighbors),
        mode: activeTab // Explicitly send 'classification' or 'regression'
      };
      
      console.log("Generating decision boundary with data:", {
        "X sample": apiData.X.slice(0, 2),
        "y sample": apiData.y.slice(0, 2),
        "y types": apiData.y.slice(0, 2).map(y => typeof y),
        "mode": apiData.mode,
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
  const resetData = () => {
    setTrainingPoints([]);
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setError(null);
    setShowDecisionBoundary(false);
    setBoundaryImg(null);
    setHoveredPoint(null);
    setNearestNeighbors([]);
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
      // Get actual canvas dimensions
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      
      // Clear the canvas
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      
      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 0.5;
      
      // Calculate step size for grid lines (16 divisions for -8 to 8 range = 1 unit per division)
      const stepX = canvasWidth / 16;
      const stepY = canvasHeight / 16;
      
      // Draw horizontal grid lines
      for (let i = 0; i <= 16; i++) {
        const y = i * stepY;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      }
      
      // Draw vertical grid lines
      for (let i = 0; i <= 16; i++) {
        const x = i * stepX;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
      }
      
      // Draw X and Y axes with dotted lines
      ctx.strokeStyle = '#9ca3af'; // Medium gray color
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); // Dotted line pattern
      
      // X-axis (horizontal line at y=0)
      const yAxisPos = canvasHeight / 2; // y=0 position
      ctx.beginPath();
      ctx.moveTo(0, yAxisPos);
      ctx.lineTo(canvasWidth, yAxisPos);
      ctx.stroke();
      
      // Y-axis (vertical line at x=0)
      const xAxisPos = canvasWidth / 2; // x=0 position
      ctx.beginPath();
      ctx.moveTo(xAxisPos, 0);
      ctx.lineTo(xAxisPos, canvasHeight);
      ctx.stroke();
      
      // Reset line dash
      ctx.setLineDash([]);
      
      // Draw axes labels
      ctx.fillStyle = '#4b5563';
      ctx.font = '12px Inter, sans-serif';
      
      // X-axis labels
      for (let i = 0; i <= 16; i += 2) {
        const x = i * stepX;
        const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
        ctx.fillText(value.toFixed(0), x - 8, canvasHeight - 5);
      }
      
      // Y-axis labels
      for (let i = 0; i <= 16; i += 2) {
        const y = i * stepY;
        const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(0), 5, y + 4);
      }
      
      // Draw training points with smaller radius (6 instead of 8)
      trainingPoints.forEach(point => {
        // Correct position calculation - Map from data coordinates to canvas coordinates
        const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        
        // Check if this point is one of the nearest neighbors
        const isNearest = nearestNeighbors.some(np => 
          np.x1 === point.x1 && np.x2 === point.x2 && np.y === point.y
        );
        
        ctx.beginPath();
        ctx.arc(x, y, isNearest ? 8 : 6, 0, Math.PI * 2); // Bigger for nearest neighbors
        
        if (activeTab === 'classification') {
          // Different color based on class for classification - compare as string
          const pointClass = point.y.toString();
          
          // Darker version for nearest neighbors
          if (pointClass === '1') {
            ctx.fillStyle = isNearest ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)'; // Blue
          } else if (pointClass === '2') {
            ctx.fillStyle = isNearest ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)'; // Red
          } else if (pointClass === '3') {
            ctx.fillStyle = isNearest ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)'; // Green
          } else {
            ctx.fillStyle = isNearest ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)'; // Gray fallback
          }
        } else {
          // For regression, use a gradient based on value
          // Normalize value to be between 0 and 1 for color gradient
          const normalizedValue = Math.min(Math.max(parseFloat(point.y) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue);
          const g = Math.round(130 + (68 - 130) * normalizedValue);
          const b = Math.round(246 + (68 - 246) * normalizedValue);
          
          // Darker for nearest neighbors
          ctx.fillStyle = isNearest 
            ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` 
            : `rgba(${r}, ${g}, ${b}, 0.7)`;
          
          // Add text label for the value
          ctx.fillStyle = '#000000';
          ctx.font = '10px Arial';
          ctx.fillText(point.y, x + 10, y - 10);
          
          // Reset fill style for the point
          ctx.fillStyle = isNearest 
            ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` 
            : `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
        
        ctx.fill();
        ctx.strokeStyle = isNearest ? '#000' : '#333';
        ctx.lineWidth = isNearest ? 2 : 1;
        ctx.stroke();
        
        // If this is a nearest neighbor, draw a line to the hovered point
        if (isNearest && hoveredPoint) {
          const hpX = ((hoveredPoint.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
          const hpY = ((scale.y.max - hoveredPoint.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
          
          ctx.beginPath();
          ctx.setLineDash([3, 3]);
          ctx.moveTo(x, y);
          ctx.lineTo(hpX, hpY);
          ctx.strokeStyle = '#333';
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
      
      // Draw points to predict (before prediction) with smaller radius
      if (predictions.length === 0) {
        predictPoints.forEach(point => {
          // Correct position calculation
          const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
          const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
          
          const isHovered = hoveredPoint && hoveredPoint.x1 === point.x1 && hoveredPoint.x2 === point.x2;
          
          ctx.beginPath();
          ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2); // Bigger if hovered
          
          // Gray color for unpredicted points
          ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';
          
          ctx.fill();
          ctx.strokeStyle = isHovered ? '#000' : '#333';
          ctx.lineWidth = isHovered ? 2 : 1;
          ctx.stroke();
        });
      }
      
      // Draw prediction results with smaller radius
      predictions.forEach(pred => {
        // Correct position calculation
        const x = ((pred.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - pred.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        
        const isHovered = hoveredPoint && hoveredPoint.x1 === pred.x1 && hoveredPoint.x2 === pred.x2;
        
        ctx.beginPath();
        ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2); // Bigger if hovered
        
        if (activeTab === 'classification') {
          // Fill color based on predicted class with flexible matching
          const predClass = pred.predictedClass.toString();
          
          // If class starts with 1 or equals 1 or 1.0, etc. -> blue
          if (predClass.startsWith('1') || Math.round(parseFloat(predClass)) === 1) {
            ctx.fillStyle = isHovered ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)'; // Blue
          } 
          // If class starts with 2 or equals 2 or 2.0, etc. -> red
          else if (predClass.startsWith('2') || Math.round(parseFloat(predClass)) === 2) {
            ctx.fillStyle = isHovered ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)'; // Red
          }
          // If class starts with 3 or equals 3 or 3.0, etc. -> green
          else if (predClass.startsWith('3') || Math.round(parseFloat(predClass)) === 3) {
            ctx.fillStyle = isHovered ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)'; // Green
          }
          // Fallback for any other values
          else {
            ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)'; // Gray
          }
        } else {
          // For regression, use a gradient based on predicted value
          const normalizedValue = Math.min(Math.max(parseFloat(pred.predictedClass) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue);
          const g = Math.round(130 + (68 - 130) * normalizedValue);
          const b = Math.round(246 + (68 - 246) * normalizedValue);
          
          // Darker if hovered
          ctx.fillStyle = isHovered 
            ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` 
            : `rgba(${r}, ${g}, ${b}, 0.7)`;
          
          // Add text label for the predicted value
          ctx.fillStyle = '#000000';
          ctx.font = '10px Arial';
          ctx.fillText(pred.predictedClass, x + 10, y - 10);
          
          // Reset fill style for the point
          ctx.fillStyle = isHovered 
            ? `rgba(${Math.max(0, r-30)}, ${Math.max(0, g-30)}, ${Math.max(0, b-30)}, 0.9)` 
            : `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
        
        // Dotted border to indicate prediction
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = isHovered ? '#000' : '#333';
        ctx.lineWidth = isHovered ? 2 : 1.5;
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }
  }, [trainingPoints, predictPoints, predictions, scale, canvasDimensions, activeTab, hoveredPoint, nearestNeighbors]);

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
        <button 
            onClick={() => navigate('/docs')} 
            style={{
                background: 'none',
                border: 'none',
                color: '#3b82f6',
                cursor: 'pointer',
                marginLeft: '8px',
                padding: '0',
                display: 'inline-flex',
                alignItems: 'center'
            }}
            title="More information"
        >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
        </button>
      </p>
      
      {backendStatus === "disconnected" && (
        <div className="backend-status error">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>
            Backend service is not responding. Please make sure the Flask server is running on port 5000.
          </span>
        </div>
      )}
      
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

      <div className="content-container" style={{ 
        width: '100%',
        maxWidth: '100%',
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{ 
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
          gap: '1.5rem',
          width: '100%',
          marginBottom: '1.5rem'
        }}>
          {/* Left column: Interactive Plot and Decision Boundary */}
          <div style={{ 
            width: '100%',
            gridColumn: '1 / 2',
            gridRow: '1 / 3',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div className="section-header">
              <h2 className="section-title">KNN {activeTab === 'classification' ? 'Classification' : 'Regression'}</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  className="sample-data-button" 
                  onClick={resetData}
                  style={{
                    backgroundColor: '#fee2e2',
                    color: '#b91c1c'
                  }}
                >
                  Reset Data
                </button>
              </div>
            </div>
            
            {error && <div className="error-message">{error}</div>}

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                Click on the graph below to add data points. Current mode: <strong>{pointsMode === 'train' ? 'Training' : 'Prediction'}</strong>
              </p>
            </div>
            
            {/* Interactive Plot */}
            <div style={{ 
              marginBottom: '1rem',
              border: '1px solid #e5e7eb', 
              borderRadius: '0.75rem', 
              overflow: 'hidden',
              position: 'relative',
              backgroundColor: '#f9fafb',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
              width: '100%',
              height: '0',
              paddingBottom: '100%' 
            }}>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0
              }}>
                <canvas 
                  ref={canvasRef}
                  width={canvasDimensions.width}
                  height={canvasDimensions.height}
                  onClick={handleCanvasClick}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  style={{ 
                    display: 'block', 
                    cursor: 'crosshair',
                    width: '100%',
                    height: '100%'
                  }}
                />
              </div>
              
              <div style={{
                position: 'absolute',
                bottom: '10px',
                right: '10px',
                padding: '4px 8px',
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderRadius: '4px',
                fontSize: '0.8rem',
                color: '#4b5563',
                pointerEvents: 'none'
              }}>
                {pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point'}
              </div>
            </div>
            
            {/* Statistics */}
            <div style={{ 
              marginTop: '0.5rem',
              marginBottom: '1rem',
              backgroundColor: '#f9fafb', 
              padding: '1rem', 
              borderRadius: '0.5rem',
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <p style={{ fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>
                Statistics:
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                <div>
                  <strong>Training Points:</strong> {trainingPoints.length}
                </div>
                <div>
                  <strong>Points to Predict:</strong> {predictPoints.length}
                </div>
                <div>
                  <strong>Predictions Made:</strong> {predictions.length}
                </div>
              </div>
            </div>
            
            {/* Decision boundary - MOVED BELOW and SMALLER */}
            {showDecisionBoundary && (
              <div style={{ 
                marginTop: '0.5rem',
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '8px', 
                border: '1px solid #e5e7eb',
                width: '100%'
              }}>
                <h3 style={{ marginBottom: '0.5rem', fontSize: '1rem', fontWeight: '500', textAlign: 'center' }}>
                  {activeTab === 'classification' ? 'Decision Regions' : 'Regression Surface'} (k={neighbors})
                </h3>
                <div style={{ 
                  width: '100%',
                  maxWidth: '500px',  
                  margin: '0 auto',
                  position: 'relative',
                  paddingBottom: '50%', 
                  border: '1px solid #e5e7eb',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  marginBottom: '0.5rem'
                }}>
                  <img 
                    src={boundaryImg}
                    alt="KNN Decision Boundary"
                    style={{ 
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                </div>
                <p style={{ 
                  color: '#6b7280', 
                  fontSize: '0.8rem', 
                  maxWidth: '500px', 
                  margin: '0 auto', 
                  textAlign: 'center',
                  lineHeight: '1.4'
                }}>
                  {activeTab === 'classification' 
                    ? 'Colored regions show how KNN divides the feature space based on training data.'
                    : 'This surface shows predicted values across the feature space.'}
                </p>
              </div>
            )}
          </div>
          
          {/* Right column: Controls and small info boxes */}
          <div style={{ 
            width: '100%',
            gridColumn: '2 / 3',
            gridRow: '1 / 2',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <h2 className="section-title">Controls & Results</h2>
            
            {/* Parameter controls */}
            <div style={{ 
              marginBottom: '1.5rem', 
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>
              
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="neighbors-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '500', 
                  color: '#4b5563',
                  fontSize: '1rem'
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
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>1 (More flexible)</span>
                  <span>10 (More stable)</span>
                </div>
              </div>
              
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>
                Interaction Mode
              </h3>
              
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                <button
                  onClick={() => setPointsMode('train')}
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb',
                    color: pointsMode === 'train' ? 'white' : '#4b5563',
                    border: 'none',
                    borderRadius: '0.5rem',
                    cursor: 'pointer',
                    fontWeight: '500',
                    flex: 1,
                    fontSize: '0.95rem'
                  }}
                >
                  Training Points
                </button>
                <button
                  onClick={() => setPointsMode('predict')}
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb',
                    color: pointsMode === 'predict' ? 'white' : '#4b5563',
                    border: 'none',
                    borderRadius: '0.5rem',
                    cursor: 'pointer',
                    fontWeight: '500',
                    flex: 1,
                    fontSize: '0.95rem'
                  }}
                >
                  Prediction Points
                </button>
              </div>
              
              {pointsMode === 'train' && activeTab === 'classification' && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <p style={{ marginBottom: '0.75rem', color: '#4b5563', fontSize: '1rem' }}>
                    Select class for training points:
                  </p>
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button
                      onClick={() => setSelectedClass('1')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '1' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)',
                        color: selectedClass === '1' ? 'white' : '#1e40af',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem'
                      }}
                    >
                      Class 1
                    </button>
                    <button
                      onClick={() => setSelectedClass('2')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '2' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)',
                        color: selectedClass === '2' ? 'white' : '#b91c1c',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem'
                      }}
                    >
                      Class 2
                    </button>
                    <button
                      onClick={() => setSelectedClass('3')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '3' ? 'rgba(34, 197, 94, 1)' : 'rgba(34, 197, 94, 0.1)',
                        color: selectedClass === '3' ? 'white' : '#15803d',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem'
                      }}
                    >
                      Class 3
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            {/* Action buttons */}
            <div style={{ 
              marginBottom: '1.5rem',
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Actions</h3>
              <button 
                onClick={predictAllPoints}
                disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 1 || predictPoints.length < 1}
                style={{
                  width: '100%',
                  backgroundColor: loading ? '#93c5fd' : '#3b82f6',
                  color: 'white',
                  padding: '0.9rem',
                  fontSize: '1.05rem',
                  fontWeight: '500',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loading ? 'wait' : 'pointer',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                  opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 1 || predictPoints.length < 1) ? 0.7 : 1,
                  marginBottom: '1.25rem'
                }}
              >
                {loading ? 'Predicting...' : 'Predict Points'}
              </button>
              
              <button 
                onClick={generateDecisionBoundary}
                disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 5}
                style={{
                  width: '100%',
                  backgroundColor: loading ? '#c4b5fd' : '#8b5cf6',
                  color: 'white',
                  padding: '0.9rem',
                  fontSize: '1.05rem',
                  fontWeight: '500',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loading ? 'wait' : 'pointer',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                  opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 5) ? 0.7 : 1
                }}
              >
                {loading ? 'Generating...' : activeTab === 'classification' ? 'Show Decision Regions' : 'Show Regression Surface'}
              </button>
            </div>
            
            {/* SMALLER: Legend */}
            <div style={{ 
              width: '100%',
              backgroundColor: 'white', 
              padding: '1rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              fontSize: '0.85rem',
              marginBottom: '1rem'
            }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>Legend</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%', 
                    backgroundColor: 'rgba(59, 130, 246, 0.7)', 
                    border: '1px solid #333' 
                  }}></div>
                  <span>Class 1</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%', 
                    backgroundColor: 'rgba(239, 68, 68, 0.7)', 
                    border: '1px solid #333' 
                  }}></div>
                  <span>Class 2</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%', 
                    backgroundColor: 'rgba(34, 197, 94, 0.7)', 
                    border: '1px solid #333' 
                  }}></div>
                  <span>Class 3</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'rgba(156, 163, 175, 0.7)',
                    border: '1px solid #333'
                  }}></div>
                  <span>Unpredicted</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'white',
                    border: '2px dashed #333'
                  }}></div>
                  <span>Predicted</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'rgba(30, 64, 175, 0.9)',
                    border: '2px solid #000'
                  }}></div>
                  <span>Neighbors</span>
                </div>
              </div>
            </div>
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
      
      {/* Loading spinner */}
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            border: '4px solid #f3f4f6',
            borderTopColor: '#3b82f6',
            borderRadius: '50%',
            width: '40px',
            height: '40px',
            animation: 'spin 1s linear infinite'
          }}></div>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}
    </motion.div>
  );
}

export default KNN;
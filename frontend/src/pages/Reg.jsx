/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import './ModelPage.css';

function Reg() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [sampleLoading, setSampleLoading] = useState(false);
  
  // Algorithm parameters
  const [alpha, setAlpha] = useState(0.01);
  const [iterations, setIterations] = useState(100);
  const [degree, setDegree] = useState(1);
  const [isHighAlpha, setIsHighAlpha] = useState(false);
  
  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const canvasWidth = 600;
  const canvasHeight = 600;
  
  // Scale for visualization - consistent with DBScan
  const scale = useMemo(() => ({
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  }), []);
  
  // Algorithm visualization states (similar to DBScan)
  const [algorithmStatus, setAlgorithmStatus] = useState("idle"); // "idle", "running", "completed"
  const [iterations_history, setIterationsHistory] = useState([]);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [isAutoAdvancing, setIsAutoAdvancing] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(500); // milliseconds between iterations
  
  // Sample data modal
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(30);
  const [sampleNoise, setSampleNoise] = useState(5.0);
  const [sampleType, setSampleType] = useState('linear'); // linear, quadratic, sinusoidal
  
  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_URL}/health`);
        setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        console.error("Backend health check failed:", err);
        setBackendStatus("disconnected");
        setError("Backend connection error: " + (err.message || "Unknown error"));
      }
    };
    
    checkBackendHealth();
  }, []);

  // Check if alpha is too high
  useEffect(() => {
    setIsHighAlpha(alpha > 0.1);
  }, [alpha]);
  
  // Auto-advance through iterations (like in DBScan)
  useEffect(() => {
    let timer;
    
    if (isAutoAdvancing && algorithmStatus === "completed") {
      if (currentIteration < iterations_history.length - 1) {
        timer = setTimeout(() => {
          setCurrentIteration(prev => prev + 1);
        }, playbackSpeed);
      } else {
        setIsAutoAdvancing(false);
      }
    }
    
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [currentIteration, iterations_history, isAutoAdvancing, playbackSpeed, algorithmStatus]);
  
  // Convert data pairs to points - extracted for reuse
  const getValidPoints = () => {
    return dataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
      .map(pair => ({
        x: parseFloat(pair.x),
        y: parseFloat(pair.y)
      }));
  };
  
  // Canvas drawing effect - updated to match DBScan style
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    drawGrid(ctx, width, height);
    
    // Get valid points
    const points = getValidPoints();
    
    // Draw data points
    drawDataPoints(ctx, points, width, height);
    
    // Draw regression curve if results exist
    if (algorithmStatus === "idle") {
      if (results && results.coefficients && results.intercept !== undefined) {
        drawRegressionLine(ctx, results.coefficients, results.intercept, results.degree || 1, width, height);
      }
    } else if (iterations_history.length > 0 && currentIteration < iterations_history.length) {
      // Draw based on the current iteration
      const iteration = iterations_history[currentIteration];
      drawRegressionLine(ctx, iteration.coefficients, iteration.intercept, iteration.degree, width, height);
      
      // Add iteration info overlay
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.font = 'bold 14px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Iteration: ${currentIteration + 1} / ${iterations_history.length}`, 10, 20);
      ctx.font = 'normal 12px Inter, sans-serif';
      ctx.fillText(`Cost: ${iteration.cost.toFixed(6)}`, 10, 40);
    }
    
    // Add subtle text indicator if no data
    if (points.length === 0) {
      ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', width / 2, height / 2);
    }
  }, [dataPairs, results, algorithmStatus, currentIteration, iterations_history]);

  // Draw grid function - match DBScan style
  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;
    
    // Calculate step size for grid lines (16 divisions for -8 to 8 range)
    const stepX = width / 16;
    const stepY = height / 16;
    
    // Draw horizontal grid lines
    for (let i = 0; i <= 16; i++) {
      const y = i * stepY;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Draw vertical grid lines
    for (let i = 0; i <= 16; i++) {
      const x = i * stepX;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Draw X and Y axes with dotted lines
    ctx.strokeStyle = '#9ca3af'; // Medium gray color
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]); // Dotted line pattern
    
    // X-axis (horizontal line at y=0)
    const yAxisPos = height / 2; // y=0 position
    ctx.beginPath();
    ctx.moveTo(0, yAxisPos);
    ctx.lineTo(width, yAxisPos);
    ctx.stroke();
    
    // Y-axis (vertical line at x=0)
    const xAxisPos = width / 2; // x=0 position
    ctx.beginPath();
    ctx.moveTo(xAxisPos, 0);
    ctx.lineTo(xAxisPos, height);
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
      ctx.fillText(value.toFixed(0), x - 8, height - 5);
    }
    
    // Y-axis labels
    for (let i = 0; i <= 16; i += 2) {
      const y = i * stepY;
      const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
      ctx.fillText(value.toFixed(0), 5, y + 4);
    }
  };

  // Helper function to draw points
  const drawDataPoints = (ctx, points, width, height) => {
    if (!points || points.length === 0) return;
    
    points.forEach(point => {
      // Convert data coordinates to screen coordinates
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
      const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  // Updated drawRegressionLine function to match DBScan style
  const drawRegressionLine = (ctx, coefficients, intercept, degree, width, height) => {
    if (!coefficients || intercept === undefined) return;
    
    // Generate points for the curve
    const numPoints = 100;
    const xMin = scale.x.min;
    const xMax = scale.x.max;
    const xStep = (xMax - xMin) / (numPoints - 1);
    
    const points = [];
    for (let i = 0; i < numPoints; i++) {
      const x = xMin + i * xStep;
      
      // Calculate polynomial value
      let y = intercept;
      for (let j = 0; j < coefficients.length; j++) {
        y += coefficients[j] * Math.pow(x, j + 1);
      }
      
      // Convert to screen coordinates
      const screenX = ((x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
      const screenY = height - ((y - scale.y.min) / (scale.y.max - scale.y.min)) * height;
      
      points.push({ x: screenX, y: screenY });
    }
    
    // Draw the curve
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 3;
    ctx.stroke();
  };

  // Convert screen coordinates to data coordinates
  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    // Calculate with scale factor for high DPI displays
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Adjust the x and y by the scale factor
    const adjustedX = x * scaleX;
    const adjustedY = y * scaleY;
    
    const dataX = scale.x.min + (adjustedX / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - (adjustedY / canvas.height) * (scale.y.max - scale.y.min);
    
    return { x: dataX, y: dataY };
  };

  // Handle canvas click to add point
  const handleCanvasClick = (e) => {
    // Allow adding points even if algorithm is completed - only block during "running" state
    if (algorithmStatus === "running") {
      return;
    }
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    console.log("Canvas clicked at data coordinates:", dataPoint);
    
    // Add point to data pairs - use existing empty pair if available
    const emptyPairIndex = dataPairs.findIndex(pair => pair.x === '' && pair.y === '');
    const newPoint = {
      x: dataPoint.x.toFixed(2),
      y: dataPoint.y.toFixed(2)
    };
    
    if (emptyPairIndex >= 0) {
      // Update existing empty pair
      const newPairs = [...dataPairs];
      newPairs[emptyPairIndex] = newPoint;
      setDataPairs(newPairs);
    } else {
      // Add new pair
      setDataPairs([...dataPairs, newPoint]);
    }
  };

  const handleAddPair = () => {
    setDataPairs([...dataPairs, { x: '', y: '' }]);
  };

  const handleRemovePair = (index) => {
    const newPairs = [...dataPairs];
    newPairs.splice(index, 1);
    
    // Ensure there's always at least one row
    if (newPairs.length === 0) {
      newPairs.push({ x: '', y: '' });
    }
    
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
  
  // Sample data generation
  const loadSampleData = () => {
    setShowSampleDataModal(true);
  };
  
  const generateSampleData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/regression/sample_data`, {
        dataset_type: sampleType,
        n_samples: sampleCount,
        noise_level: sampleNoise / 10 // Scale down to match backend expectation
      });
      
      if (response.data.X && response.data.y) {
        // Convert sample data into pairs
        const pairs = response.data.X.map((x, index) => ({
          x: x.toString(),
          y: response.data.y[index].toString()
        }));
        
        setDataPairs(pairs);
      } else {
        setError("Failed to generate sample data: No data points received");
      }
      
      setShowSampleDataModal(false);
    } catch (err) {
      console.error("Error generating sample data:", err);
      setError(`Failed to generate sample data: ${err.message || "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setResults(null);
    setIterationsHistory([]);
    setCurrentIteration(0);
    setAlgorithmStatus("idle");
    setIsAutoAdvancing(false);
    setError(null);
  };

  // Update the algorithm status handling to allow re-running with different parameters
  const handleRunModel = async () => {
    // Filter out empty pairs
    const validPairs = getValidPoints();
    
    // Validate inputs
    if (validPairs.length < 2) {
      setError('Please add at least 2 data points for meaningful regression');
      return;
    }

    setError(null);
    setLoading(true);
    // Don't reset results before new request - this allows seeing previous results during loading
    setAlgorithmStatus("running");

    try {
      // Prepare data for the API
      const targetRecordPoints = 100;
      const step_interval = iterations > targetRecordPoints ? 
        Math.max(1, Math.floor(iterations / targetRecordPoints)) : 1;
      const apiData = {
        X: validPairs.map(pair => pair.x),
        y: validPairs.map(pair => pair.y),
        alpha: alpha,
        iterations: iterations,
        degree: degree,
        record_steps: true, // Request iteration history
        step_interval: step_interval
      };

      console.log('Calling API endpoint with data:', apiData);
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/regression`, apiData);
      
      if (response.data.error) {
        throw new Error(response.data.error);
      }
      
      console.log('Results state set to:', response.data);
      setResults(response.data);
      
      if (response.data.iteration_history && response.data.iteration_history.length > 0) {
        setIterationsHistory(response.data.iteration_history);
        setCurrentIteration(0);
        setAlgorithmStatus("completed");
      } else {
        setAlgorithmStatus("idle");
      }
    } catch (err) {
      console.error('Error details:', err);
      setAlgorithmStatus("idle");
      
      // Check if the error is related to learning rate
      const errorMessage = err.message || 'An error occurred while running the model.';
      
      if (
        errorMessage.toLowerCase().includes('diverg') || 
        errorMessage.toLowerCase().includes('explod') ||
        errorMessage.toLowerCase().includes('inf') ||
        errorMessage.toLowerCase().includes('nan') ||
        errorMessage.toLowerCase().includes('overflow')
      ) {
        setError(
          `Error: Gradient descent failed to converge. This is likely due to a learning rate (${alpha.toFixed(4)}) that is too high. ` +
          `Try reducing the learning rate or using a lower polynomial degree.`
        );
      } else {
        setError(`Error: ${errorMessage} Please try again.`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Playback control methods (like in DBScan)
  const goToStart = () => {
    setCurrentIteration(0);
  };
  
  const goToEnd = () => {
    setCurrentIteration(iterations_history.length - 1);
  };
  
  const stepForward = () => {
    if (currentIteration < iterations_history.length - 1) {
      setCurrentIteration(prev => prev + 1);
    }
  };
  
  const stepBackward = () => {
    if (currentIteration > 0) {
      setCurrentIteration(prev => prev - 1);
    }
  };
  
  const goToIteration = (index) => {
    if (index >= 0 && index < iterations_history.length) {
      setCurrentIteration(index);
    }
  };
  
  const togglePlayback = () => {
    setIsAutoAdvancing(prev => !prev);
  };
  
  const SampleDataModal = () => (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'white',
        padding: '2rem',
        borderRadius: '0.5rem',
        maxWidth: '500px',
        width: '90%'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1.5rem'
        }}>
          <h2 style={{
            fontSize: '1.5rem',
            fontWeight: '600',
            margin: 0
          }}>
            Generate Sample Data
          </h2>
          <button 
            onClick={() => setShowSampleDataModal(false)}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              color: '#6b7280'
            }}
          >
            ×
          </button>
        </div>
        
        {/* Dataset Type Selection */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            fontWeight: '500', 
            color: '#4b5563' 
          }}>
            Dataset Type
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            <button
              onClick={() => setSampleType('linear')}
              style={{
                padding: '0.5rem 0.75rem',
                backgroundColor: sampleType === 'linear' ? '#3b82f6' : '#e5e7eb',
                color: sampleType === 'linear' ? 'white' : '#4b5563',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.9rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <span style={{ fontWeight: '500' }}>Linear</span>
              <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Straight line</span>
            </button>
            
            <button
              onClick={() => setSampleType('quadratic')}
              style={{
                padding: '0.5rem 0.75rem',
                backgroundColor: sampleType === 'quadratic' ? '#3b82f6' : '#e5e7eb',
                color: sampleType === 'quadratic' ? 'white' : '#4b5563',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.9rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <span style={{ fontWeight: '500' }}>Quadratic</span>
              <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Parabolic curve</span>
            </button>
            
            <button
              onClick={() => setSampleType('sinusoidal')}
              style={{
                padding: '0.5rem 0.75rem',
                backgroundColor: sampleType === 'sinusoidal' ? '#3b82f6' : '#e5e7eb',
                color: sampleType === 'sinusoidal' ? 'white' : '#4b5563',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.9rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <span style={{ fontWeight: '500' }}>Sinusoidal</span>
              <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Wave pattern</span>
            </button>
          </div>
        </div>
        
        {/* Number of Samples */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="samples-slider" style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            fontWeight: '500', 
            color: '#4b5563' 
          }}>
            Number of Samples: {sampleCount}
          </label>
          <input
            id="samples-slider"
            type="range"
            min="10"
            max="100"
            step="5"
            value={sampleCount}
            onChange={(e) => setSampleCount(Number(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            fontSize: '0.8rem', 
            color: '#6b7280',
            marginTop: '0.25rem'
          }}>
            <span>10 (Fewer points)</span>
            <span>100 (More points)</span>
          </div>
        </div>
        
        {/* Noise level */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="noise-slider" style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            fontWeight: '500', 
            color: '#4b5563' 
          }}>
            Noise Level: {sampleNoise.toFixed(1)}
          </label>
          <input
            id="noise-slider"
            type="range"
            min="1.0"
            max="10.0"
            step="0.5"
            value={sampleNoise}
            onChange={(e) => setSampleNoise(Number(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            fontSize: '0.8rem', 
            color: '#6b7280',
            marginTop: '0.25rem'
          }}>
            <span>1.0 (Clean data)</span>
            <span>10.0 (Noisy data)</span>
          </div>
        </div>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'flex-end',
          gap: '0.75rem',
          marginTop: '1.5rem'
        }}>
          <button
            onClick={() => setShowSampleDataModal(false)}
            style={{
              padding: '0.6rem 1.2rem',
              backgroundColor: '#f3f4f6',
              color: '#4b5563',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Cancel
          </button>
          <button
            onClick={generateSampleData}
            style={{
              padding: '0.6rem 1.2rem',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Generate
          </button>
        </div>
      </div>
    </div>
  );

  // Page title and description
  const pageTitle = "Polynomial Regression";
  const pageDescription = "Polynomial Regression extends linear regression to fit an nth degree polynomial equation to data, modeling non-linear relationships.";

  return (
    <motion.div 
      className="model-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {showSampleDataModal && <SampleDataModal />}
      
      <div className="model-header">
        <button className="back-button" onClick={() => navigate('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
        </button>
        <h1 className="model-title">{pageTitle}</h1>
      </div>

      <p className="model-description">
        {pageDescription}
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

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

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
          <div style={{ 
            width: '100%',
            gridColumn: '1 / 2',
            gridRow: '1 / 2',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div className="section-header" style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1rem'
            }}>
              <h2 className="section-title">Data Points</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  className="sample-data-button" 
                  onClick={loadSampleData}
                  disabled={loading || algorithmStatus !== "idle"}
                  style={{
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.375rem',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    fontWeight: '500',
                    opacity: loading || algorithmStatus !== "idle" ? 0.7 : 1
                  }}
                >
                  Load Sample Data
                </button>
                <button 
                  className="reset-button" 
                  onClick={resetData}
                  disabled={loading || dataPairs.length <= 1 && dataPairs[0].x === '' && dataPairs[0].y === ''}
                  style={{
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#ef4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.375rem',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    fontWeight: '500',
                    opacity: loading || (dataPairs.length <= 1 && dataPairs[0].x === '' && dataPairs[0].y === '') ? 0.7 : 1
                  }}
                >
                  Reset Data
                </button>
              </div>
            </div>
            
            <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>
              {algorithmStatus === "running" ? 
                "Polynomial regression is running. Please wait..." : 
                algorithmStatus === "completed" ?
                "Use the playback controls below to see how the regression line evolves over iterations." :
                "Click on the canvas below to add data points. You can adjust parameters and run the regression when ready."}
            </p>

            <div style={{ 
              marginBottom: '1rem',
              border: '1px solid #e5e7eb', 
              borderRadius: '0.75rem', 
              overflow: 'hidden',
              position: 'relative',
              backgroundColor: '#f9fafb',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
              width: '100%',
              height: 0,
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
                  width={canvasWidth}
                  height={canvasHeight}
                  onClick={handleCanvasClick}
                  style={{ 
                    display: 'block', 
                    cursor: algorithmStatus === "idle" ? 'crosshair' : 'default',
                    width: '100%',
                    height: '100%'
                  }}
                />
              </div>
              
              {/* Click overlay hint */}
              {algorithmStatus === "idle" && (
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
                  Click to add point
                </div>
              )}
            </div>
            
            {algorithmStatus === "completed" && iterations_history.length > 0 && (
              <div className="playback-controls" style={{
                backgroundColor: 'white',
                padding: '1rem',
                borderRadius: '0.5rem',
                border: '1px solid #e5e7eb',
                marginBottom: '1rem'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  gap: '1rem', 
                  marginBottom: '0.75rem' 
                }}>
                  <button
                    onClick={goToStart}
                    disabled={currentIteration === 0}
                    style={{
                      padding: '0.5rem',
                      backgroundColor: 'transparent',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: 'pointer',
                      color: currentIteration === 0 ? '#d1d5db' : '#374151'
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="19 20 9 12 19 4 19 20"></polygon>
                      <line x1="5" y1="19" x2="5" y2="5"></line>
                    </svg>
                  </button>
                  
                  <button
                    onClick={stepBackward}
                    disabled={currentIteration === 0}
                    style={{
                      padding: '0.5rem',
                      backgroundColor: 'transparent',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: 'pointer',
                      color: currentIteration === 0 ? '#d1d5db' : '#374151'
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="19 20 9 12 19 4 19 20"></polygon>
                    </svg>
                  </button>
                  
                  <button
                    onClick={togglePlayback}
                    style={{
                      padding: '0.75rem',
                      backgroundColor: '#3b82f6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '50%',
                      width: '40px',
                      height: '40px',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}
                  >
                    {isAutoAdvancing ? (
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="6" y="4" width="4" height="16"></rect>
                        <rect x="14" y="4" width="4" height="16"></rect>
                      </svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                      </svg>
                    )}
                  </button>
                  
                  <button
                    onClick={stepForward}
                    disabled={currentIteration === iterations_history.length - 1}
                    style={{
                      padding: '0.5rem',
                      backgroundColor: 'transparent',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: 'pointer',
                      color: currentIteration === iterations_history.length - 1 ? '#d1d5db' : '#374151'
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: 'rotate(180deg)' }}>
                      <polygon points="19 20 9 12 19 4 19 20"></polygon>
                    </svg>
                  </button>
                  
                  <button
                    onClick={goToEnd}
                    disabled={currentIteration === iterations_history.length - 1}
                    style={{
                      padding: '0.5rem',
                      backgroundColor: 'transparent',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: 'pointer',
                      color: currentIteration === iterations_history.length - 1 ? '#d1d5db' : '#374151'
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: 'rotate(180deg)' }}>
                      <polygon points="19 20 9 12 19 4 19 20"></polygon>
                      <line x1="5" y1="19" x2="5" y2="5"></line>
                    </svg>
                  </button>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ position: 'relative', width: '100%', marginBottom: '1.5rem' }}>
                    <div style={{ 
                      width: '100%',
                      height: '8px',
                      backgroundColor: '#e5e7eb',
                      borderRadius: '4px',
                      position: 'relative',
                      cursor: 'pointer'
                    }} 
                    onClick={(e) => {
                      // Calculate position relative to the bar
                      const rect = e.currentTarget.getBoundingClientRect();
                      const x = e.clientX - rect.left;
                      const percentage = x / rect.width;
                      // Calculate the iteration based on position
                      const newIteration = Math.floor(percentage * iterations_history.length);
                      goToIteration(newIteration);
                    }}>
                      <div 
                        style={{ 
                          position: 'absolute',
                          top: '0',
                          left: '0',
                          width: `${(currentIteration / Math.max(1, iterations_history.length - 1)) * 100}%`,
                          height: '100%',
                          backgroundColor: '#3b82f6',
                          borderRadius: '4px',
                          transition: 'width 0.3s ease'
                        }}
                      ></div>
                    </div>
                    
                    <span style={{ 
                      position: 'absolute',
                      top: '-20px',
                      left: `${(currentIteration / Math.max(1, iterations_history.length - 1)) * 100}%`,
                      transform: 'translateX(-50%)',
                      backgroundColor: '#3b82f6',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      fontWeight: '500'
                    }}>
                      {currentIteration + 1} of {iterations_history.length}
                    </span>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ fontSize: '0.8rem', color: '#6b7280', whiteSpace: 'nowrap' }}>Speed:</span>
                    <input 
                      type="range"
                      min="500"
                      max="4000"
                      step="50"
                      value={4050 - playbackSpeed}
                      onChange={(e) => setPlaybackSpeed(4050 - parseInt(e.target.value))}
                      style={{ flex: 1 }}
                    />
                    <span style={{ fontSize: '0.8rem', color: '#374151' }}>
                      {playbackSpeed < 500 ? 'Fast' : 'Slow'}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
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
                  <strong>Data Points:</strong> {getValidPoints().length}
                </div>
                <div>
                  <strong>Polynomial Degree:</strong> {degree}
                </div>
                <div>
                  <strong>Learning Rate:</strong> {alpha.toFixed(4)}
                </div>
              </div>
            </div>
            
            {/* Results section moved here */}
            {results && (
              <div style={{ 
                backgroundColor: 'white', 
                padding: '1.5rem', 
                borderRadius: '6px', 
                border: '1px solid #e5e7eb',
                width: '100%',
                marginBottom: '1rem'
              }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Results</h3>
                
                <div style={{
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  padding: '1rem',
                  backgroundColor: '#f9fafb',
                  marginBottom: '1rem'
                }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>Model Equation</div>
                  <div style={{ fontWeight: '600', fontSize: '1.2rem', fontFamily: 'math, serif' }}>
                    {`y = ${
                      results.intercept !== undefined ? 
                      (results.intercept >= 0 ? results.intercept.toFixed(4) : results.intercept.toFixed(4)) : 
                      '?'
                    } ${
                      results.coefficients ? 
                      results.coefficients.map((coef, index) => 
                        `${coef >= 0 ? '+ ' : '- '}${Math.abs(coef).toFixed(4)}x${index+1 > 1 ? `^${index+1}` : ''}`
                      ).join(' ') : 
                      ''
                    }`}
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                  <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                    <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>R² Score</div>
                    <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                      {results.r2 !== undefined ? results.r2.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                  
                  <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                    <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Mean Squared Error</div>
                    <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                      {results.mse !== undefined ? results.mse.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                </div>
                
                {/* Model insights section */}
                <div style={{ backgroundColor: 'white', padding: '0.5rem 0' }}>
                  <p style={{ fontSize: '0.9rem', color: '#4b5563', marginBottom: '0.5rem' }}>
                    {degree === 1 ? (
                      <>
                        Linear regression (degree 1) fits a straight line to the data. 
                        {results && results.r2 > 0.8 ? (
                          " Your data appears to have a strong linear relationship."
                        ) : results && results.r2 > 0.5 ? (
                          " Your data shows a moderate linear trend."
                        ) : (
                          " Your data might benefit from a higher polynomial degree to capture non-linear patterns."
                        )}
                      </>
                    ) : degree <= 3 ? (
                      <>
                        Polynomial regression with degree {degree} can model curved relationships in the data.
                        {results && results.r2 > 0.8 ? (
                          " This model fits your data well."
                        ) : results && results.r2 > 0.5 ? (
                          " The model captures some patterns in your data."
                        ) : (
                          " Consider adjusting the polynomial degree or adding more data points."
                        )}
                      </>
                    ) : (
                      <>
                        Higher-degree polynomials (degree {degree}) can fit complex curves but may overfit with limited data.
                        {results && results.r2 > 0.9 ? (
                          " While the R² score is high, be cautious of overfitting."
                        ) : (
                          " If performance is poor, try reducing the degree or adding more data points."
                        )}
                      </>
                    )}
                  </p>
                </div>
              </div>
            )}
          </div>
          
          <div style={{ 
            width: '100%',
            gridColumn: '2 / 3',
            gridRow: '1 / 2',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <h2 className="section-title" style={{ marginBottom: '1rem' }}>Algorithm Controls</h2>
            
            <div style={{ 
              marginBottom: '1.5rem', 
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Parameters</h3>
              
              {/* Learning Rate Slider */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="alpha-param" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  Learning Rate (Alpha): {alpha.toFixed(4)}
                </label>
                <input 
                  id="alpha-param"
                  type="range"
                  min="0.0001"
                  max="0.9"
                  step="0.0001"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  disabled={algorithmStatus === "running"}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>0.0001 (Slow learning)</span>
                  <span>0.9 (Fast learning)</span>
                </div>
               
              </div>
              
              {/* Iterations Slider */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="iterations-param" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  Max Iterations: {iterations}
                </label>
                <input 
                  id="iterations-param"
                  type="range"
                  min="1"
                  max="1000"
                  step="1"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value))}
                  disabled={algorithmStatus === "running"}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>1 (Faster)</span>
                  <span>1000 (More accurate)</span>
                </div>
              </div>
              
              {/* Polynomial Degree Slider */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="degree-param" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  Polynomial Degree: {degree}
                </label>
                <input 
                  id="degree-param"
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={degree}
                  onChange={(e) => setDegree(parseInt(e.target.value))}
                  disabled={algorithmStatus === "running"}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>1 (Linear)</span>
                  <span>10 (Higher-order)</span>
                </div>
              </div>
              
              <div style={{ marginTop: '2rem' }}>
                <button
                  onClick={handleRunModel}
                  disabled={loading || backendStatus === "disconnected" || getValidPoints().length < 2 || algorithmStatus === "running"}
                  style={{
                    width: '100%',
                    padding: '0.75rem 1rem',
                    backgroundColor: loading ? '#93c5fd' : '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.5rem',
                    cursor: 'pointer',
                    fontWeight: '500',
                    fontSize: '1rem',
                    opacity: (loading || backendStatus === "disconnected" || getValidPoints().length < 2 || algorithmStatus === "running") ? 0.7 : 1
                  }}
                >
                  {loading ? 'Running...' : 'Run Polynomial Regression'}
                </button>
              </div>
            </div>

            {/* Cost history plot moved here */}
            {results && results.cost_history_plot && (
              <div style={{ 
                marginBottom: '1.5rem', 
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '6px', 
                border: '1px solid #e5e7eb',
                width: '100%'
              }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Curve</h3>
                <img 
                  src={`data:image/png;base64,${results.cost_history_plot}`} 
                  alt="Cost History Plot" 
                  style={{ width: '100%', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}
                />
                <p style={{ color: '#6b7280', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  This graph shows how the Mean Squared Error decreases over iterations during training.
                </p>
              </div>
            )}
            
            
          </div>
        </div>
      </div>
      {showSampleDataModal && <SampleDataModal />}
    </motion.div>
  );
}

export default Reg;
/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef } from 'react';
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
  const [alpha, setAlpha] = useState(0.01);
  const [iterations, setIterations] = useState(100);
  const [isHighAlpha, setIsHighAlpha] = useState(false);
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(30);
  const [sampleNoise, setSampleNoise] = useState(5.0);
  
  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 550, height: 400 });
  // Updated scale to center the graph at the origin and set axes from -8 to 8
  const scale = {
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  };
  
  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
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

  // Check if alpha is too high
  useEffect(() => {
    setIsHighAlpha(alpha > 0.1);
  }, [alpha]);
  
  // Convert data pairs to points - extracted for reuse
  const getValidPoints = () => {
    return dataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
      .map(pair => ({
        x: parseFloat(pair.x),
        y: parseFloat(pair.y)
      }));
  };
  
  // Draw canvas with points and regression line
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid and axes
    drawGrid(ctx, canvas);
    
    // Get valid points
    const points = getValidPoints();
    
    // Draw points
    drawPoints(ctx, canvas, points);
    
    // Draw regression line if results exist
    if (results && results.coefficient !== undefined && results.intercept !== undefined) {
      drawRegressionLine(ctx, canvas, results.coefficient, results.intercept);
    }
    
    // Debug console logs
    console.log("Canvas redrawn with:", {
      points: points.length,
      results: results ? "yes" : "no"
    });
    
  }, [dataPairs, results]);

  // Updated drawGrid function to reflect the new scale
  const drawGrid = (ctx, canvas) => {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    // Draw horizontal grid lines
    for (let i = scale.y.min; i <= scale.y.max; i++) {
      const y = canvas.height - ((i - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw vertical grid lines
    for (let i = scale.x.min; i <= scale.x.max; i++) {
      const x = ((i - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;

    // X-axis
    const xAxisY = canvas.height - ((0 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, xAxisY);
    ctx.lineTo(canvas.width, xAxisY);
    ctx.stroke();

    // Y-axis
    const yAxisX = ((0 - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
    ctx.beginPath();
    ctx.moveTo(yAxisX, 0);
    ctx.lineTo(yAxisX, canvas.height);
    ctx.stroke();

    // Draw axes labels
    ctx.fillStyle = '#4b5563';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';

    // X-axis labels
    for (let i = scale.x.min; i <= scale.x.max; i += 2) {
      const x = ((i - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      ctx.fillText(i.toString(), x - 10, xAxisY + 15);
    }

    // Y-axis labels
    for (let i = scale.y.min; i <= scale.y.max; i += 2) {
      const y = canvas.height - ((i - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      ctx.fillText(i.toString(), yAxisX + 5, y + 4);
    }
  };

  // Helper function to draw points
  const drawPoints = (ctx, canvas, points) => {
    if (!points || points.length === 0) return;
    
    points.forEach(point => {
      // Convert data coordinates to screen coordinates
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  // Helper function to draw regression line
  const drawRegressionLine = (ctx, canvas, coefficient, intercept) => {
    // Don't try to draw a line if we don't have valid coefficient/intercept
    if (coefficient === undefined || intercept === undefined) return;
    
    ctx.beginPath();
    
    // Calculate line endpoints in data coordinates
    const x1 = scale.x.min;
    const y1 = coefficient * x1 + intercept;
    const x2 = scale.x.max;
    const y2 = coefficient * x2 + intercept;
    
    // Convert to screen coordinates
    const sx1 = 0;
    const sy1 = canvas.height - ((y1 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
    const sx2 = canvas.width;
    const sy2 = canvas.height - ((y2 - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
    
    ctx.moveTo(sx1, sy1);
    ctx.lineTo(sx2, sy2);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 3;
    ctx.stroke();
  };

  // Convert screen coordinates to data coordinates
  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.min + ((canvas.height - y) / canvas.height) * (scale.y.max - scale.y.min);
    
    return { x: dataX, y: dataY };
  };

  // Handle canvas click to add point
  const handleCanvasClick = (e) => {
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

  const generateSampleDataWithOptions = async () => {
    setSampleLoading(true);
    setShowSampleDataModal(false);
    setError(null);

    try {
      const sampleData = await getLinearRegressionSampleData(sampleCount, sampleNoise);

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

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setResults(null);
    setError(null);
  };

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
    setResults(null); // Reset results before new request

    try {
      // Prepare data for the API
      const apiData = {
        X: validPairs.map(pair => pair.x),
        y: validPairs.map(pair => pair.y),
        alpha: alpha,
        iterations: iterations
      };

      console.log('Calling API endpoint with data:', apiData);
      const response = await runLinearRegression(apiData);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      console.log('Results state set to:', response);
      setResults(response);
    } catch (err) {
      console.error('Error details:', err);
      
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
          `Try reducing the learning rate to a value below 1.0.`
        );
      } else {
        setError(`Error: ${errorMessage} Please try again.`);
      }
    } finally {
      setLoading(false);
    }
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
        padding: '1.5rem',
        borderRadius: '8px',
        width: '90%',
        maxWidth: '500px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem'
        }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '600' }}>
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

        {/* Sample count slider */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="sample-count" style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            fontWeight: '500', 
            color: '#4b5563' 
          }}>
            Number of Samples: {sampleCount}
          </label>
          <input
            id="sample-count"
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

        {/* Noise level slider */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label htmlFor="noise" style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            fontWeight: '500', 
            color: '#4b5563' 
          }}>
            Noise Level: {sampleNoise.toFixed(1)}
          </label>
          <input
            id="noise"
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
            onClick={generateSampleDataWithOptions}
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
            Generate Data
          </button>
        </div>
      </div>
    </div>
  );

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
        Linear Regression is a supervised learning algorithm that models the relationship between 
        a dependent variable and one or more independent variables using a linear equation.
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

      <div className="content-container" style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '2rem',
        alignItems: 'start'
      }}>
        <div className="input-section">
          <div className="section-header">
            <h2 className="section-title">Interactive Data Input</h2>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button 
                className="sample-data-button" 
                onClick={() => setShowSampleDataModal(true)} 
                disabled={sampleLoading}
              >
                {sampleLoading ? 'Loading...' : 'Generate Sample Data'}
              </button>
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

          <div style={{ marginBottom: '1.5rem' }}>
            <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
              Click on the graph below to add data points, or manually enter values in the table.
            </p>
          </div>
          
          {/* Interactive Canvas */}
          <div style={{ 
            marginBottom: '1.5rem',
            border: '1px solid #e5e7eb', 
            borderRadius: '0.75rem', 
            overflow: 'hidden',
            position: 'relative',
            backgroundColor: '#f9fafb',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
          }}>
            <canvas 
              ref={canvasRef}
              width={canvasDimensions.width}
              height={canvasDimensions.height}
              onClick={handleCanvasClick}
              style={{ 
                display: 'block', 
                cursor: 'crosshair'
              }}
            />
            
            {/* Add a click overlay hint */}
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
          </div>
          
          <h3 style={{ fontSize: '1.1rem', marginBottom: '0.75rem' }}>Data Points Table</h3>
          <div style={{ 
            marginBottom: '1.5rem',
            maxHeight: '200px', 
            overflowY: 'auto',
            border: '1px solid #e5e7eb',
            borderRadius: '0.5rem',
            padding: '0.5rem'
          }}>
            <div style={{ marginBottom: '0.5rem', display: 'flex', fontWeight: 'bold' }}>
              <div style={{ flex: 1 }}>X</div>
              <div style={{ flex: 1 }}>Y</div>
              <div style={{ width: '30px' }}></div>
            </div>
            
            {dataPairs.map((pair, index) => (
              <div key={index} className="data-input-container" style={{
                display: 'flex',
                alignItems: 'center',
                marginBottom: '5px'
              }}>
                <div style={{ display: 'flex', flex: 1, gap: '5px' }}>
                  <input
                    type="text"
                    style={{
                      flex: 1,
                      padding: '5px 8px',
                      border: '1px solid #d1d5db',
                      borderRadius: '4px'
                    }}
                    placeholder="X value"
                    value={pair.x}
                    onChange={(e) => handleInputChange(index, 'x', e.target.value)}
                  />
                  <input
                    type="text"
                    style={{
                      flex: 1,
                      padding: '5px 8px',
                      border: '1px solid #d1d5db',
                      borderRadius: '4px'
                    }}
                    placeholder="Y value"
                    value={pair.y}
                    onChange={(e) => handleInputChange(index, 'y', e.target.value)}
                  />
                </div>
                <button 
                  onClick={() => handleRemovePair(index)}
                  aria-label="Remove data point"
                  style={{
                    width: '28px',
                    height: '28px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f3f4f6',
                    border: 'none',
                    borderRadius: '4px',
                    marginLeft: '8px',
                    cursor: 'pointer',
                    fontSize: '1.2rem',
                    color: '#6b7280'
                  }}
                >
                  ×
                </button>
              </div>
            ))}
            
            <button 
              onClick={handleAddPair}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                padding: '5px',
                backgroundColor: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                marginTop: '10px',
                color: '#4b5563',
                fontWeight: '500'
              }}
            >
              + Add Data Point
            </button>
          </div>
          
          {/* Learning Rate Slider */}
          <div style={{ marginTop: '1.5rem' }}>
            <label htmlFor="alpha-slider" style={{ 
              display: 'block', 
              marginBottom: '0.5rem', 
              fontWeight: '500', 
              color: '#4b5563' 
            }}>
              Learning Rate (Alpha): {alpha.toFixed(4)}
            </label>
            <input
              id="alpha-slider"
              type="range"
              min="0.0001"
              max="0.12"
              step="0.0001"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              fontSize: '0.8rem', 
              color: '#6b7280',
              marginTop: '0.25rem'
            }}>
              <span>0.0001 (Slow learning)</span>
              <span>0.12 (Fast learning)</span>
            </div>
            
            {isHighAlpha && (
              <div style={{
                marginTop: '0.5rem',
                padding: '0.5rem',
                backgroundColor: '#FEF2F2',
                borderRadius: '0.375rem',
                borderLeft: '3px solid #EF4444',
                color: '#B91C1C',
                fontSize: '0.875rem'
              }}>
                <strong>Warning:</strong> High learning rates may cause exploding gradients and divergence. 
                Values below 0.1 are recommended for most datasets.
              </div>
            )}
          </div>
          
          {/* Iterations Slider */}
          <div style={{ marginTop: '1.5rem' }}>
            <label htmlFor="iterations-slider" style={{ 
              display: 'block', 
              marginBottom: '0.5rem', 
              fontWeight: '500', 
              color: '#4b5563' 
            }}>
              Max Iterations: {iterations}
            </label>
            <input
              id="iterations-slider"
              type="range"
              min="1"
              max="500"
              step="1"
              value={iterations} 
              onChange={(e) => setIterations(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              fontSize: '0.8rem', 
              color: '#6b7280',
              marginTop: '0.25rem'
            }}>
              <span>1 (Fast but might not converge)</span>
              <span>500 (More accurate)</span>
            </div>
          </div>
          
          <div style={{ marginTop: '1.5rem' }}>
            <button 
              onClick={handleRunModel}
              disabled={loading || backendStatus === "disconnected"}
              style={{
                width: '100%',
                backgroundColor: loading ? '#93c5fd' : '#3b82f6',
                color: 'white',
                padding: '12px',
                fontSize: '1.1rem',
                fontWeight: '500',
                border: 'none',
                borderRadius: '6px',
                cursor: loading ? 'wait' : 'pointer',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1
              }}
            >
              {loading ? 'Running...' : 'Run Regression'}
            </button>
          </div>
        </div>

        <div className="results-section">
          <h2 className="section-title">Results</h2>
          {loading ? (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '200px'
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
          ) : results ? (
            <div style={{ padding: '1rem' }}>
              <div style={{
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                padding: '1rem',
                backgroundColor: 'white',
                marginBottom: '1rem'
              }}>
                <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>Model Equation</div>
                <div style={{ fontWeight: '600', fontSize: '1.2rem' }}>
                  {`y = ${results.coefficient !== undefined ? results.coefficient.toFixed(4) : '?'} × x + ${results.intercept !== undefined ? results.intercept.toFixed(4) : '?'}`}
                </div>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Coefficient</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {results.coefficient !== undefined ? results.coefficient.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Intercept</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {results.intercept !== undefined ? results.intercept.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>R² Score</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {results.r2 !== undefined ? results.r2.toFixed(4) : 'N/A'}
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Mean Squared Error</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {results.mse !== undefined ? results.mse.toFixed(4) : 'N/A'}
                  </div>
                </div>
              </div>
              
              {/* Show cost history plot if available */}
              {results.cost_history_plot && (
                <div style={{ marginTop: '1.5rem', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Learning Curve:</h3>
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
              
              {/* Model Fit Visualization Info - note that actual visualization is shown on the canvas */}
              <div style={{ 
                marginTop: '1.5rem', 
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '6px', 
                border: '1px solid #e5e7eb' 
              }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Model Fit Visualization:</h3>
                <p style={{ color: '#6b7280', fontSize: '0.9rem' }}>
                  The regression line is shown in red on the interactive graph. Blue points are your data.
                </p>
              </div>
            </div>
          ) : (
            <div style={{ 
              color: '#6b7280', 
              textAlign: 'center', 
              padding: '2rem',
              backgroundColor: '#f9fafb',
              borderRadius: '6px',
              border: '1px dashed #d1d5db' 
            }}>
              <p>Add points by clicking on the interactive graph or using the table.</p>
              <p>Then click "Run Regression" to see results.</p>
            </div>
          )}
        </div>
      </div>
      {showSampleDataModal && <SampleDataModal />}
    </motion.div>
  );
}

export default LinReg;
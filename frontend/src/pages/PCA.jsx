/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { runPCA, getPCASampleData, checkHealth } from '../api';
import './ModelPage.css';

function PCA() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const safeDataPairs = dataPairs || [];
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [sampleLoading, setSampleLoading] = useState(false);
  const [components, setComponents] = useState(2);
  const [explainedVariance, setExplainedVariance] = useState(null);
  const [showSampleDataModal, setShowSampleDataModal] = useState(false);
  const [sampleCount, setSampleCount] = useState(30);
  const [sampleNoise, setSampleNoise] = useState(5.0);
  const [projectedPoints, setProjectedPoints] = useState([]); // New state for projected points
  
  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 600, height: 600 });
  const [isProjectionActive, setIsProjectionActive] = useState(false);
  // Fixed scale - no adjustment
  const scale = {
    x: { min: -5, max: 5 },
    y: { min: -5, max: 5 }
  };
  
  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await checkHealth();
        setBackendStatus(response.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        console.error("Backend health check failed:", err);
        setBackendStatus("disconnected");
        setError("Backend connection error: " + (err.message || "Unknown error"));
      }
    };
    
    checkBackendHealth();
  }, []);
  
  // Convert data pairs to points - extracted for reuse
  const getValidPoints = () => {
    return safeDataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
      .map(pair => ({
        x: parseFloat(pair.x),
        y: parseFloat(pair.y)
      }));
  };
  
  // Draw canvas with points and PCA components
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

    // Draw projected points
    if (projectedPoints.length > 0) {
      drawPoints(ctx, canvas, projectedPoints, 'rgba(239, 51, 18, 0.7)'); // Red color for projected points
    }

    // Draw principal components if results exist
    if (results && results.components && results.original_mean) {
        drawPrincipalComponents(ctx, canvas, results.components, results.original_mean, points);
    }
  }, [dataPairs, projectedPoints, results, isProjectionActive]);

  // Helper function to draw grid and axes
  const drawGrid = (ctx, canvas) => {
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
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
    
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
  };

  // Helper function to draw points
  const drawPoints = (ctx, canvas, points, color = 'rgba(59, 130, 246, 0.7)') => {
    if (!points || points.length === 0) return;
    
    points.forEach(point => {
      // Convert data coordinates to screen coordinates
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color; // Color for points
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  // Helper function to draw principal components
  const drawPrincipalComponents = (ctx, canvas, components, originalMean, points) => {
    const meanX = ((originalMean[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
    const meanY = canvas.height - ((originalMean[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
  
    const pc1 = components[0];
    const pc2 = components[1];
  
    // Scale factor for drawing principal components
    const scaleFactor = Math.min(canvas.width, canvas.height) * 0.4;
  
    // Draw PCA1 (First Principal Component)
    const pc1x = meanX + pc1[0] * scaleFactor;
    const pc1y = meanY - pc1[1] * scaleFactor;
    ctx.beginPath();
    ctx.moveTo(meanX, meanY);
    ctx.lineTo(pc1x, pc1y);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
    ctx.stroke();
    drawArrow(ctx, pc1x, pc1y, pc1[0], pc1[1], 10); // Draw arrowhead for PCA1
  
    // Draw PCA2 (Second Principal Component)
    const pc2x = meanX + pc2[0] * scaleFactor;
    const pc2y = meanY - pc2[1] * scaleFactor;
    ctx.beginPath();
    ctx.moveTo(meanX, meanY);
    ctx.lineTo(pc2x, pc2y);
    ctx.strokeStyle = 'indigo';
    ctx.lineWidth = 3;
    ctx.stroke();
    drawArrow(ctx, pc2x, pc2y, pc2[0], pc2[1], 10); // Draw arrowhead for PCA2
  
    // Draw the mean point
    ctx.beginPath();
    ctx.arc(meanX, meanY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(251, 191, 36, 0.7)';
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
  
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillStyle = 'black';
    ctx.fillText('PC1', pc1x + 5, pc1y - 5);
    ctx.fillStyle = 'indigo';
    ctx.fillText('PC2', pc2x + 5, pc2y - 5);
    ctx.fillStyle = 'rgba(251, 191, 36, 1)';
    ctx.fillText('Mean', meanX + 10, meanY);
  
    // Draw projections if active
    if (isProjectionActive) {
      points.forEach(point => {
        const dx = point.x - originalMean[0];
        const dy = point.y - originalMean[1];
  
        // Project point onto PCA1
        const projectionLength = dx * pc1[0] + dy * pc1[1];
        const projectedX = originalMean[0] + projectionLength * pc1[0];
        const projectedY = originalMean[1] + projectionLength * pc1[1];
  
        // Convert data coordinates to screen coordinates
        const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        const projX = ((projectedX - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const projY = canvas.height - ((projectedY - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
  
        // Draw dotted line from original point to projected point
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.moveTo(x, y);
        ctx.lineTo(projX, projY);
        ctx.strokeStyle = 'rgba(107, 114, 128, 0.7)';
        ctx.stroke();
        ctx.setLineDash([]);
  
        // Draw projected point
        ctx.beginPath();
        ctx.arc(projX, projY, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 51, 18, 0.7)';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }
  };
  
  // Helper function to draw arrows
  const drawArrow = (ctx, x, y, dx, dy, size) => {
    // Normalize direction vector
    const length = Math.sqrt(dx*dx + dy*dy);
    const dirX = dx / length;
    const dirY = dy / length;
    
    // Calculate perpendicular vector
    const perpX = -dirY;
    const perpY = dirX;
    
    // Draw arrow head
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - size * dirX + size * 0.5 * perpX, y + size * dirY - size * 0.5 * perpY);
    ctx.lineTo(x - size * dirX - size * 0.5 * perpX, y + size * dirY + size * 0.5 * perpY);
    ctx.closePath();
    ctx.fillStyle = "black";
    ctx.fill();
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
    const emptyPairIndex = safeDataPairs.findIndex(pair => pair.x === '' && pair.y === '');
    const newPoint = {
      x: dataPoint.x.toFixed(2),
      y: dataPoint.y.toFixed(2)
    };
    
    if (emptyPairIndex >= 0) {
      // Update existing empty pair
      const newPairs = [...safeDataPairs];
      newPairs[emptyPairIndex] = newPoint;
      setDataPairs(newPairs);
    } else {
      // Add new pair
      setDataPairs([...safeDataPairs, newPoint]);
    }
  };

  const handleAddPair = () => {
    setDataPairs([...safeDataPairs, { x: '', y: '' }]);
  };

  const handleRemovePair = (index) => {
    const newPairs = [...safeDataPairs];
    newPairs.splice(index, 1);
    
    // Ensure there's always at least one row
    if (newPairs.length === 0) {
      newPairs.push({ x: '', y: '' });
    }
    
    setDataPairs(newPairs);
  };

  const handleProjectPoints = () => {
    if (!results || !results.components || !results.original_mean) {
      console.error("No results available for projecting points.");
      return;
    }

    const pc1 = results.components[0]; // First principal component
    const mean = results.original_mean; // Mean vector

    // Calculate projections for each point
    const projectedPairs = safeDataPairs
      .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(pair.x) && !isNaN(pair.y)) // Filter valid points
      .map(pair => {
        const x = parseFloat(pair.x);
        const y = parseFloat(pair.y);

        // Calculate projection onto PCA1
        const dx = x - mean[0];
        const dy = y - mean[1];
        const projectionLength = dx * pc1[0] + dy * pc1[1];
        const projectedX = mean[0] + projectionLength * pc1[0];
        const projectedY = mean[1] + projectionLength * pc1[1];

        return {
          x: projectedX.toFixed(2), // Round to 2 decimal places
          y: projectedY.toFixed(2)
        };
      });

    setProjectedPoints(projectedPairs); // Store projected points separately
    setIsProjectionActive(true);
  };

  const handleInputChange = (index, field, value) => {
    const newPairs = [...safeDataPairs];
    
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
      const sampleData = await getPCASampleData();

      // Convert sample data into pairs
      const pairs = sampleData.X.map((point) => ({
        x: point[0]?.toString() || '',
        y: point[1]?.toString() || ''
      }));

      setDataPairs(pairs);
    } catch (err) {
      setError('Failed to load sample data. Please try again.');
      console.error(err);
    } finally {
      setSampleLoading(false);
    }
  };

  const generateSampleDataWithOptions = async () => {
    setSampleLoading(true);
    setShowSampleDataModal(false);
    setError(null);

    try {
      // Pass slider values dynamically to the API call
      const sampleData = await getPCASampleData(sampleCount, sampleNoise);

      // Convert sample data into pairs
      const pairs = sampleData.X.map((point) => ({
        x: point[0]?.toString() || '',
        y: point[1]?.toString() || ''
      }));

      setDataPairs(pairs);
    } catch (err) {
      setError('Failed to load sample data. Please try again.');
      console.error(err);
    } finally {
      setSampleLoading(false);
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

  const resetData = () => {
    setDataPairs([{ x: '', y: '' }]);
    setResults(null);
    setError(null);
    setProjectedPoints([]); // Clear projected points
    setIsProjectionActive(false); 
  };

  const handleRunModel = async () => {
    console.log('Running PCA...');
    setIsProjectionActive(false); // Reset projection state

    // Clear the canvas explicitly
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      console.log('Canvas cleared');
    }

    // Reset dataPairs and projected points
    setDataPairs((prevPairs) => prevPairs.filter(pair => pair.x !== '' && pair.y !== ''));
    setProjectedPoints([]); // Clear projected points

    const validPoints = getValidPoints();
    console.log('Valid points:', validPoints);

    if (validPoints.length < 2) {
      setError('Please add at least 2 data points for meaningful PCA');
      return;
    }

    setError(null);
    setLoading(true);
    setResults(null);

    try {
      const apiData = {
        X: validPoints.map(point => [point.x, point.y])
      };

      console.log('Sending data to PCA API:', apiData);
      const response = await runPCA(apiData);

      if (response.error) {
        throw new Error(response.error);
      }

      console.log('PCA API response:', response);
      setResults(response);
    } catch (err) {
      console.error('PCA Error details:', err);
      setError(`Error: ${err.message || 'An unknown error occurred'}. Please try again.`);
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
        <h1 className="model-title">Principal Component Analysis (PCA)</h1>
      </div>

      <p className="model-description">
        Principal Component Analysis (PCA) is a dimensionality reduction technique that identifies the directions (principal components) 
        along which the data varies the most. This visualization reduces 2D data to 1D, showing the principal components and data projection.
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
            
            {safeDataPairs.map((pair, index) => (
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
          
          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
  <button 
    onClick={handleRunModel}
    disabled={loading || backendStatus === "disconnected"}
    style={{
      flex: 1,
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
    {loading ? 'Running...' : 'Compute PCA'}
  </button>
  <button 
    onClick={handleProjectPoints}
    disabled={!results || loading || backendStatus === "disconnected"}
    style={{
      flex: 1,
      backgroundColor: '#8b5cf6',
      color: 'white',
      padding: '12px',
      fontSize: '1.1rem',
      fontWeight: '500',
      border: 'none',
      borderRadius: '6px',
      cursor: !results ? 'not-allowed' : 'pointer',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
      opacity: !results ? 0.7 : 1
    }}
  >
    Projecting Points
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
                <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>Principal Components</div>
                <div style={{ fontWeight: '500', fontSize: '1.1rem', display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: 'rgb(0, 0, 0)' }}>PC1: [{results.components[0][0].toFixed(4)}, {results.components[0][1].toFixed(4)}]</span>
                  <span style={{ color: 'rgb(12, 12, 12)' }}>PC2: [{results.components[1][0].toFixed(4)}, {results.components[1][1].toFixed(4)}]</span>
                </div>
              </div>

              <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.5rem' }}>Eigenvalues and Eigenvectors</div>
              <div style={{ fontWeight: '500', fontSize: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div>
                  <span style={{ color: 'rgb(16, 16, 16)' }}>Eigenvalue 1: {results.explained_variance[0].toFixed(4)}</span>
                  <br />
                  <span style={{ color: 'rgb(3, 3, 3)' }}>Eigenvector 1: [{results.components[0][0].toFixed(4)}, {results.components[0][1].toFixed(4)}]</span>
                </div>
                <div>
                  <span style={{ color: 'rgb(5, 5, 5)' }}>Eigenvalue 2: {results.explained_variance[1].toFixed(4)}</span>
                  <br />
                  <span style={{ color: 'rgb(0, 0, 0)' }}>Eigenvector 2: [{results.components[1][0].toFixed(4)}, {results.components[1][1].toFixed(4)}]</span>
                </div>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Explained Variance Ratio (PC1)</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {(results.explained_variance_ratio[0] * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Explained Variance Ratio (PC2)</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {(results.explained_variance_ratio[1] * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Mean Vector</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    [{results.original_mean[0].toFixed(4)}, {results.original_mean[1].toFixed(4)}]
                  </div>
                </div>
                
                <div style={{ padding: '1rem', backgroundColor: 'white', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <div style={{ fontWeight: '500', color: '#6b7280', marginBottom: '0.25rem' }}>Cumulative Variance</div>
                  <div style={{ fontWeight: '600', fontSize: '1.1rem' }}>
                    {((results.explained_variance_ratio[0] + results.explained_variance_ratio[1]) * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
              
              {/* Show heatmap if available */}
              {results.corr_heatmap && (
                <div style={{ marginTop: '1.5rem', backgroundColor: 'white', padding: '1rem', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
                  <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>Correlation Matrix:</h3>
                  <img 
                    src={`data:image/png;base64,${results.corr_heatmap}`} 
                    alt="Correlation Heatmap" 
                    style={{ width: '100%', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}
                  />
                  <p style={{ color: '#6b7280', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                    Correlation heatmap showing relationships between variables.
                  </p>
                </div>
              )}
              
              {/* Original vs Reconstructed Visualization Info */}
              <div style={{ 
                marginTop: '1.5rem', 
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '6px', 
                border: '1px solid #e5e7eb' 
              }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>PCA Visualization Guide:</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(59, 130, 246, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Blue points: Original data points</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(239, 68, 68, 0.5)', borderRadius: '50%', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Red points: Projected data (if using 1D projection)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(251, 191, 36, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Yellow point: Data mean (center)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '3px', backgroundColor: 'rgb(1, 1, 1)', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Black line: First principal component  (PC1)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '3px', backgroundColor: 'rgba(139, 92, 246, 1)', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Purple line: Second principal component (PC2)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '3px', borderBottom: '2px dashed rgba(107, 114, 128, 0.7)', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Dotted lines: Projection lines between original and reconstructed points</span>
                  </div>
                </div>
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
              <p>Then click "Compute PCA" to see results.</p>
            </div>
          )}
        </div>
      </div>
      {showSampleDataModal && <SampleDataModal />}
    </motion.div>
  );
}

export default PCA;
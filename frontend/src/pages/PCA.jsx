import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { runPCA, getPCASampleData, checkHealth } from '../api';
import './ModelPage.css';
const API_URL = 'http://localhost:5000/api';

function PCA() {
  const navigate = useNavigate();
  const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [sampleLoading, setSampleLoading] = useState(false);
  
  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 550, height: 400 });
  // Fixed scale - no adjustment
  const scale = {
    x: { min: -5, max: 5 },
    y: { min: -5, max: 5 }
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
  
  // Convert data pairs to points - extracted for reuse
  const getValidPoints = () => {
    return dataPairs
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
    
    // Draw principal components if results exist
    if (results && results.components && results.mean) {
      drawPrincipalComponents(ctx, canvas, results.components, results.mean, points);
      
      // Draw reconstructed points if available
      if (results.reconstructed) {
        drawReconstructedPoints(ctx, canvas, results.reconstructed);
      }
    }
    
    // Debug console logs
    console.log("Canvas redrawn with:", {
      points: points.length,
      results: results ? "yes" : "no"
    });
    
  }, [dataPairs, results]);

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
  const drawPoints = (ctx, canvas, points) => {
    if (!points || points.length === 0) return;
    
    points.forEach(point => {
      // Convert data coordinates to screen coordinates
      const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point.y - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue color for original points
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  // Helper function to draw reconstructed points
  const drawReconstructedPoints = (ctx, canvas, reconstructedPoints) => {
    if (!reconstructedPoints || reconstructedPoints.length === 0) return;
    
    reconstructedPoints.forEach(point => {
      // Convert data coordinates to screen coordinates
      const x = ((point[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
      const y = canvas.height - ((point[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.5)'; // Red color for reconstructed points
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Draw dotted line connecting to original points (if results.original exists)
      if (results && results.original) {
        const originalPoint = results.original.find(
          (op, idx) => reconstructedPoints.indexOf(point) === idx
        );
        
        if (originalPoint) {
          const x2 = ((originalPoint[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
          const y2 = canvas.height - ((originalPoint[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
          
          ctx.beginPath();
          ctx.setLineDash([2, 2]);
          ctx.moveTo(x, y);
          ctx.lineTo(x2, y2);
          ctx.strokeStyle = 'rgba(107, 114, 128, 0.7)'; // Gray color for connecting lines
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    });
  };

  // Helper function to draw principal components
  const drawPrincipalComponents = (ctx, canvas, components, mean, points) => {
    const meanX = ((mean[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
    const meanY = canvas.height - ((mean[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
    
    // Calculate scaling factor based on points spread
    let maxDist = 0;
    points.forEach(point => {
      const dx = point.x - mean[0];
      const dy = point.y - mean[1];
      const dist = Math.sqrt(dx*dx + dy*dy);
      maxDist = Math.max(maxDist, dist);
    });
    
    const scaleFactor = Math.min(canvas.width, canvas.height) * 0.4 / (maxDist || 1);
    
    // Draw PC1 (first principal component)
    const pc1 = components[0];
    const pc1x = meanX + pc1[0] * scaleFactor;
    const pc1y = meanY - pc1[1] * scaleFactor;
    
    ctx.beginPath();
    ctx.moveTo(meanX - pc1[0] * scaleFactor, meanY + pc1[1] * scaleFactor);
    ctx.lineTo(pc1x, pc1y);
    ctx.strokeStyle = 'rgba(16, 185, 129, 1)'; // Green color for PC1
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Add arrow to PC1
    drawArrow(ctx, pc1x, pc1y, pc1[0], pc1[1], 10);
    
    // Draw PC2 (second principal component)
    const pc2 = components[1];
    const pc2x = meanX + pc2[0] * scaleFactor;
    const pc2y = meanY - pc2[1] * scaleFactor;
    
    ctx.beginPath();
    ctx.moveTo(meanX - pc2[0] * scaleFactor, meanY + pc2[1] * scaleFactor);
    ctx.lineTo(pc2x, pc2y);
    ctx.strokeStyle = 'rgba(139, 92, 246, 1)'; // Purple color for PC2
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Add arrow to PC2
    drawArrow(ctx, pc2x, pc2y, pc2[0], pc2[1], 10);
    
    // Draw mean point
    ctx.beginPath();
    ctx.arc(meanX, meanY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(251, 191, 36, 0.7)'; // Yellow color for mean
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Add text labels
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillStyle = 'rgba(16, 185, 129, 1)';
    ctx.fillText('PC1', pc1x + 5, pc1y - 5);
    
    ctx.fillStyle = 'rgba(139, 92, 246, 1)';
    ctx.fillText('PC2', pc2x + 5, pc2y - 5);
    
    ctx.fillStyle = 'rgba(251, 191, 36, 1)';
    ctx.fillText('Mean', meanX + 10, meanY);
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

  const loadSampleData = async () => {
    setSampleLoading(true);
    setError(null);
    
    try {
      const sampleData = await getPCASampleData();
      
      // Convert sample data into pairs
      const pairs = sampleData.X.map((point) => ({
        x: point[0].toString(),
        y: point[1].toString()
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

  // Update the handleRunModel function for better error logging
  const handleRunModel = async () => {
    // Filter out empty pairs
    const validPoints = getValidPoints();
    
    // Validate inputs
    if (validPoints.length < 2) {
      setError('Please add at least 2 data points for meaningful PCA');
      return;
    }

    setError(null);
    setLoading(true);
    setResults(null); // Reset results before new request

    try {
      // Prepare data for the API
      const apiData = {
        X: validPoints.map(point => [point.x, point.y])
      };

      console.log('Calling PCA API with URL:', `${API_URL}/pca`);
      console.log('Calling PCA API with data:', apiData);
      
      const response = await runPCA(apiData);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      console.log('PCA Results received:', response);
      setResults(response);
    } catch (err) {
      console.error('PCA Error details:', err);
      
      let errorMessage = 'An unknown error occurred';
      
      if (err.message && err.message.includes('Network Error')) {
        errorMessage = 'Network error: Cannot connect to the server. Please check if the backend is running and verify the endpoint URLs.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(`Error: ${errorMessage}. Please try again.`);
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
                onClick={loadSampleData}
                disabled={sampleLoading}
              >
                {sampleLoading ? 'Loading...' : 'Load Sample Data'}
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
              {loading ? 'Running...' : 'Compute PCA'}
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
                  <span style={{ color: 'rgba(16, 185, 129, 1)' }}>PC1: [{results.components[0][0].toFixed(4)}, {results.components[0][1].toFixed(4)}]</span>
                  <span style={{ color: 'rgba(139, 92, 246, 1)' }}>PC2: [{results.components[1][0].toFixed(4)}, {results.components[1][1].toFixed(4)}]</span>
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
                    [{results.mean[0].toFixed(4)}, {results.mean[1].toFixed(4)}]
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
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Red points: Reconstructed data (if using 1D projection)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', backgroundColor: 'rgba(251, 191, 36, 0.7)', borderRadius: '50%', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Yellow point: Data mean (center)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '3px', backgroundColor: 'rgba(16, 185, 129, 1)', display: 'inline-block' }}></span>
                    <span style={{ color: '#4b5563', fontSize: '0.9rem' }}>Green line: First principal component (PC1)</span>
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
    </motion.div>
  );
}

export default PCA;
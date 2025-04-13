import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import './ModelPage.css';

function KMeans() {
  const navigate = useNavigate();
  const [data, setData] = useState({ X: [] });
  const [clusterCount, setClusterCount] = useState(3);
  const [maxIterations, setMaxIterations] = useState(100);
  const [sampleCount, setSampleCount] = useState(100);
  const [sampleClusters, setSampleClusters] = useState(3);
  const [clusterVariance, setClusterVariance] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");
  const [datasetType, setDatasetType] = useState('blobs');
  const [canvasBounds] = useState({ xMin: -8, xMax: 8, yMin: -8, yMax: 8 });
  const [animationRunning, setAnimationRunning] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  
  // For clustering results
  const [clusters, setClusters] = useState(null);
  const [centroids, setCentroids] = useState([]);
  const [clusterLabels, setClusterLabels] = useState([]);
  
  // Canvas ref and state for interactive plotting
  const canvasRef = useRef(null);
  const [canvasDimensions] = useState({ width: 600, height: 600 }); // Square canvas for proper aspect ratio
  
  // Define colors for clusters (up to 10 clusters)
  const clusterColors = [
    'rgba(59, 130, 246, 0.7)',  // Blue
    'rgba(239, 68, 68, 0.7)',   // Red
    'rgba(34, 197, 94, 0.7)',   // Green
    'rgba(168, 85, 247, 0.7)',  // Purple
    'rgba(251, 146, 60, 0.7)',  // Orange
    'rgba(236, 72, 153, 0.7)',  // Pink
    'rgba(14, 165, 233, 0.7)',  // Sky blue
    'rgba(234, 179, 8, 0.7)',   // Yellow
    'rgba(8, 145, 178, 0.7)',   // Cyan
    'rgba(124, 58, 237, 0.7)'   // Indigo
  ];
  
  const apiUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000';

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${apiUrl}/health`);
        setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        console.error("Backend health check failed:", err);
        setBackendStatus("disconnected");
        setError("Backend connection error: " + (err.message || "Unknown error"));
      }
    };
    
    checkBackendHealth();
  }, []);
  
  // Display notification as an error message
  const showNotification = (message, type = 'success') => {
    setError(type === 'error' ? message : null);
    
    // Clear error after 3 seconds if it's a success message
    if (type === 'success') {
      setTimeout(() => {
        setError(null);
      }, 3000);
    }
  };
  
  // Generate sample data for clustering
  const generateSampleData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get(
        `${apiUrl}/kmeans/sample?n_samples=${sampleCount}&n_clusters=${sampleClusters}&variance=${clusterVariance}&dataset_type=${datasetType}`
      );
      
      setData(response.data);
      
      // Reset any clustering results
      setClusters(null);
      setCentroids([]);
      setClusterLabels([]);
      setCurrentIteration(0);
      
      showNotification('Sample data generated successfully');
    } catch (err) {
      console.error('Error generating sample data:', err);
      showNotification('Failed to generate sample data', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Reset the data
  const resetData = () => {
    setData({ X: [] });
    setClusters(null);
    setCentroids([]);
    setClusterLabels([]);
    setCurrentIteration(0);
    setError(null);
  };

  // Handle canvas click to add points
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    // Add point to data
    const newData = {
      ...data,
      X: [...(data.X || []), [dataPoint.x, dataPoint.y]]
    };
    
    setData(newData);
    
    // Reset clustering results when adding new points
    setClusters(null);
    setCentroids([]);
    setClusterLabels([]);
    setCurrentIteration(0);
  };
  
  // Convert screen coordinates to data coordinates
  const screenToData = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const scaleFactorX = canvas.width / rect.width;
    const scaleFactorY = canvas.height / rect.height;
    
    // Adjust the x and y by the scale factor
    const adjustedX = x * scaleFactorX;
    const adjustedY = y * scaleFactorY;
    
    // Now convert to data coordinates
    const dataX = canvasBounds.xMin + (adjustedX / canvas.width) * (canvasBounds.xMax - canvasBounds.xMin);
    const dataY = canvasBounds.yMax - (adjustedY / canvas.height) * (canvasBounds.yMax - canvasBounds.yMin);
    
    return { x: dataX, y: dataY };
  };

  const runKMeans = async () => {
    try {
      if (!data.X || data.X.length < clusterCount) {
        showNotification(`Please add at least ${clusterCount} data points for ${clusterCount} clusters`, 'error');
        return;
      }
  
      setLoading(true);
      setError(null);
      setAnimationRunning(true);
      setCurrentIteration(0);
      
      const requestData = {
        X: data.X,
        k: clusterCount,
        max_iterations: maxIterations,
        return_history: true  // Request iteration history
      };
  
      const response = await axios.post(`${apiUrl}/kmeans`, requestData);
      
      if (response.data.error) {
        throw new Error(response.data.error);
      }
      
      // Animate through iterations
      const history = response.data.history || [];
      const totalSteps = history.length;
      
      // Start animation
      let step = 0;
      const animateStep = () => {
        if (step < totalSteps) {
          const currentStep = history[step];
          setCentroids(currentStep.centroids);
          setClusterLabels(currentStep.labels);
          setCurrentIteration(currentStep.iteration);
          
          // Show every iteration with a 500ms delay (0.5 second)
          step = step + 1;
          setTimeout(animateStep, 1000);
        } else {
          // Animation finished
          setAnimationRunning(false);
          setClusters({
            centroids: response.data.centroids,
            labels: response.data.labels,
            inertia: response.data.inertia,
            silhouette_score: response.data.silhouette_score,
            iterations: response.data.iterations
          });
        }
      };
      
      // Start the animation
      animateStep();
      
    } catch (err) {
      console.error('Error running K-means clustering:', err);
      showNotification(`Failed to run K-means clustering: ${err.message}`, 'error');
      setAnimationRunning(false);
    } finally {
      setLoading(false);
    }
  };

  // Draw canvas with grid, axes, points, and clusters
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    drawGrid(ctx, canvas);
    
    // Draw axes
    drawAxes(ctx, canvas);
    
    // Draw data points
    if (data && data.X && data.X.length > 0) {
      drawPoints(ctx, canvas, data.X, clusterLabels, centroids);
    } else {
      // Add subtle text indicator if no data
      ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', width / 2, height / 2);
    }
    
  }, [data, canvasDimensions, canvasBounds, clusterLabels, centroids, currentIteration]);
  
  // Helper function to draw grid
  const drawGrid = (ctx, canvas) => {
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;
    
    // Calculate step size for grid lines (16 divisions)
    const stepX = canvas.width / 16;
    const stepY = canvas.height / 16;
    
    // Draw horizontal grid lines
    for (let i = 0; i <= 16; i++) {
      const y = i * stepY;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
    
    // Draw vertical grid lines
    for (let i = 0; i <= 16; i++) {
      const x = i * stepX;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
  };
  
  // Helper function to draw axes
  const drawAxes = (ctx, canvas) => {
    // Calculate step size for grid lines (needed for labels)
    const stepX = canvas.width / 16;
    const stepY = canvas.height / 16;
    
    // X and Y axes with dotted lines
    ctx.strokeStyle = '#9ca3af'; // Medium gray color
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]); // Dotted line pattern
    
    // X-axis (horizontal line at y=0)
    const yAxisPos = canvas.height / 2; // y=0 position
    ctx.beginPath();
    ctx.moveTo(0, yAxisPos);
    ctx.lineTo(canvas.width, yAxisPos);
    ctx.stroke();
    
    // Y-axis (vertical line at x=0)
    const xAxisPos = canvas.width / 2; // x=0 position
    ctx.beginPath();
    ctx.moveTo(xAxisPos, 0);
    ctx.lineTo(xAxisPos, canvas.height);
    ctx.stroke();
    
    // Reset line dash
    ctx.setLineDash([]);
    
    // Draw axes labels
    ctx.fillStyle = '#4b5563';
    ctx.font = '12px Inter, sans-serif';
    
    // X-axis labels
    for (let i = 0; i <= 16; i += 2) {
      const x = i * stepX;
      const value = canvasBounds.xMin + (i / 16) * (canvasBounds.xMax - canvasBounds.xMin);
      ctx.fillText(value.toFixed(0), x - 8, canvas.height - 5);
    }
    
    // Y-axis labels
    for (let i = 0; i <= 16; i += 2) {
      const y = i * stepY;
      const value = canvasBounds.yMax - (i / 16) * (canvasBounds.yMax - canvasBounds.yMin);
      ctx.fillText(value.toFixed(0), 5, y + 4);
    }
  };
  
  // Inside the runKMeans function, update the animation logic:


// Inside the drawPoints function, update the centroid drawing:
const drawPoints = (ctx, canvas, points, labels, centroids) => {
  const { width, height } = canvas;
  const { xMin, xMax, yMin, yMax } = canvasBounds;
  
  // Draw points
  points.forEach((point, index) => {
    // Convert data coordinates to screen coordinates
    const x = ((point[0] - xMin) / (xMax - xMin)) * width;
    const y = height - ((point[1] - yMin) / (yMax - yMin)) * height;
    
    // Determine point color based on cluster label (if available)
    let fillColor;
    if (labels && labels.length > index) {
      const clusterIndex = labels[index];
      fillColor = clusterColors[clusterIndex % clusterColors.length];
    } else {
      fillColor = 'rgba(107, 114, 128, 0.7)'; // Default gray if no cluster assigned
    }
    
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  
  // Draw centroids if available
  if (centroids && centroids.length > 0) {
    centroids.forEach((centroid, index) => {
      // Convert data coordinates to screen coordinates
      const x = ((centroid[0] - xMin) / (xMax - xMin)) * width;
      const y = height - ((centroid[1] - yMin) / (yMax - yMin)) * height;
      
      const clusterColor = clusterColors[index % clusterColors.length];
      
      // Draw centroid as a star
      const radius = 8;
      const spikes = 5;
      const outerRadius = radius;
      const innerRadius = radius / 2;
      
      ctx.beginPath();
      let rot = Math.PI / 2 * 3;
      let x2 = x;
      let y2 = y;
      const step = Math.PI / spikes;
      
      ctx.moveTo(x2, y2 - outerRadius);
      
      for (let i = 0; i < spikes; i++) {
        x2 = x + Math.cos(rot) * outerRadius;
        y2 = y + Math.sin(rot) * outerRadius;
        ctx.lineTo(x2, y2);
        rot += step;
        
        x2 = x + Math.cos(rot) * innerRadius;
        y2 = y + Math.sin(rot) * innerRadius;
        ctx.lineTo(x2, y2);
        rot += step;
      }
      
      ctx.lineTo(x, y - outerRadius);
      ctx.closePath();
      
      // Fill with cluster color
      ctx.fillStyle = clusterColor;
      ctx.fill();
      
      // Add darker border
      ctx.strokeStyle = '#000'; // Black border
      ctx.lineWidth = 2; // Thicker border line
      ctx.stroke();
    });
    
    // Draw current iteration info
    if (currentIteration > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(`Iteration: ${currentIteration}`, 10, 20);
    }
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
        <h1 className="model-title">K-Means Clustering</h1>
      </div>

      <p className="model-description">
        K-means clustering is an unsupervised learning algorithm that partitions data into K clusters,
        where each data point belongs to the cluster with the nearest mean.
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

      <div style={{ 
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
          {/* Left column: Interactive Plot */}
          <div style={{ 
            width: '100%',
            gridColumn: '1 / 2',
            gridRow: '1 / 3',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div className="section-header">
              <h2 className="section-title">Interactive Clustering</h2>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  className="sample-data-button" 
                  onClick={generateSampleData}
                  disabled={loading}
                >
                  {loading ? 'Loading...' : 'Generate Sample Data'}
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

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                Click on the graph below to add data points manually, or use the "Generate Sample Data" button.
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
                Click to add point
              </div>
              
              {/* Iteration info overlay */}
              {animationRunning && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  padding: '6px 12px',
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  borderRadius: '4px',
                  fontSize: '0.9rem',
                  color: 'white',
                  pointerEvents: 'none'
                }}>
                  Running iteration: {currentIteration}
                </div>
              )}
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
                  <strong>Data Points:</strong> {data.X ? data.X.length : 0}
                </div>
                {clusters && (
                  <>
                    <div>
                      <strong>Clusters:</strong> {clusterCount}
                    </div>
                    <div>
                      <strong>Iterations:</strong> {clusters.iterations}
                    </div>
                  </>
                )}
              </div>
            </div>
            
            {/* Legend & How K-Means Works - MOVED BELOW */}
            <div style={{ 
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Legend</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>
                {/* Show first 6 cluster colors */}
                {clusterColors.slice(0, 6).map((color, index) => (
                  <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ 
                      width: '12px', 
                      height: '12px', 
                      borderRadius: '50%', 
                      backgroundColor: color, 
                      border: '1px solid #333' 
                    }}></div>
                    <span>Cluster {index + 1}</span>
                  </div>
                ))}
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'rgba(107, 114, 128, 0.7)',
                    border: '1px solid #333'
                  }}></div>
                  <span>Unclustered</span>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="14" height="14" viewBox="0 0 100 100">
                    <polygon 
                      points="50,5 63,38 100,38 69,59 82,96 50,75 18,96 31,59 0,38 37,38" 
                      fill="#3b82f6" 
                      stroke="#000" 
                      strokeWidth="5"
                    />
                  </svg>
                  <span>Centroid</span>
                </div>
              </div>
              
              <h3 style={{ marginBottom: '0.75rem', fontSize: '1.1rem', fontWeight: '500' }}>How K-Means Works</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>
                  K-Means clusters data in these steps:
                </p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>Initialize K centroids randomly</li>
                  <li>Assign points to nearest centroid</li>
                  <li>Move centroids to cluster means</li>
                  <li>Repeat until convergence</li>
                </ol>
                <p style={{ fontSize: '0.8rem', fontStyle: 'italic' }}>
                  The interactive plot shows centroids (stars) moving through iterations.
                </p>
              </div>
            </div>
          </div>
          
          {/* Right column: Controls and info boxes */}
          <div style={{ 
            width: '100%',
            gridColumn: '2 / 3',
            gridRow: '1 / 2',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <h2 className="section-title">Controls & Results</h2>
            
            {/* Dataset Options Panel */}
            <div style={{ 
              marginBottom: '1.5rem', 
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Dataset Options</h3>
              
              <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  Dataset Type
                </label>
                <select 
                  value={datasetType}
                  onChange={(e) => setDatasetType(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.6rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '0.375rem',
                    backgroundColor: 'white'
                  }}
                >
                  <option value="blobs">Isotropic Blobs</option>
                  <option value="moons">Two Moons</option>
                  <option value="circles">Concentric Circles</option>
                </select>
              </div>
              
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
                  min="50"
                  max="500"
                  step="50"
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
                  <span>50 (Fewer points)</span>
                  <span>500 (More points)</span>
                </div>
              </div>
              
              <div style={{ marginBottom: '1.25rem' }}>
                <label htmlFor="clusters-gen-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  True Clusters for Generation: {sampleClusters}
                </label>
                <input
                  id="clusters-gen-slider"
                  type="range"
                  min="2"
                  max="8"
                  step="1"
                  value={sampleClusters}
                  onChange={(e) => setSampleClusters(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.8rem', 
                  color: '#6b7280',
                  marginTop: '0.25rem'
                }}>
                  <span>2 clusters</span>
                  <span>8 clusters</span>
                </div>
              </div>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label htmlFor="variance-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563' 
                }}>
                  Cluster Variance: {clusterVariance.toFixed(1)}
                </label>
                <input
                  id="variance-slider"
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={clusterVariance}
                  onChange={(e) => setClusterVariance(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.8rem', 
                  color: '#6b7280',
                  marginTop: '0.25rem'
                }}>
                  <span>0.1 (Tight clusters)</span>
                  <span>2.0 (Spread out)</span>
                </div>
              </div>
            </div>
            
            {/* K-Means Parameters */}
            <div style={{ 
              marginBottom: '1.5rem', 
              backgroundColor: 'white', 
              padding: '1.5rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Clustering Parameters</h3>
              
              <div style={{ marginBottom: '1.25rem' }}>
                <label htmlFor="clusters-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563',
                  fontSize: '1rem'
                }}>
                  Number of Clusters (K): {clusterCount}
                </label>
                <input
                  id="clusters-slider"
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={clusterCount}
                  onChange={(e) => setClusterCount(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>2 clusters</span>
                  <span>10 clusters</span>
                </div>
              </div>
              
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="iterations-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem', 
                  fontWeight: '500', 
                  color: '#4b5563',
                  fontSize: '1rem'
                }}>
                  Max Iterations: {maxIterations}
                </label>
                <input
                  id="iterations-slider"
                  type="range"
                  min="10"
                  max="300"
                  step="10"
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>10 iterations</span>
                  <span>300 iterations</span>
                </div>
              </div>
              
              <button 
                onClick={runKMeans}
                disabled={loading || backendStatus === "disconnected" || !data.X || data.X.length < clusterCount}
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
                  opacity: (loading || backendStatus === "disconnected" || !data.X || data.X.length < clusterCount) ? 0.7 : 1,
                  marginBottom: '1.25rem'
                }}
              >
                {loading ? 'Running...' : 'Run K-Means Clustering'}
              </button>
            </div>
            
            {/* Results Panel */}
            {clusters && (
              <div style={{ 
                width: '100%',
                backgroundColor: 'white', 
                padding: '1.5rem', 
                borderRadius: '6px', 
                border: '1px solid #e5e7eb',
                fontSize: '0.95rem'
              }}>
                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Clustering Results</h3>
                
                <div style={{ marginBottom: '1rem' }}>
                  <h4 style={{ color: '#4b5563', fontWeight: '500', marginBottom: '0.25rem' }}>Inertia: {clusters.inertia.toFixed(2)}</h4>
                  <p style={{ color: '#6b7280', fontSize: '0.85rem' }}>
                    Sum of squared distances to nearest centroid. Lower values indicate tighter clusters.
                  </p>
                </div>
                
                {clusters.silhouette_score !== undefined && (
                  <div style={{ marginBottom: '1rem' }}>
                    <h4 style={{ color: '#4b5563', fontWeight: '500', marginBottom: '0.25rem' }}>
                      Silhouette Score: {clusters.silhouette_score.toFixed(4)}
                    </h4>
                    <p style={{ color: '#6b7280', fontSize: '0.85rem' }}>
                      Measures how similar points are to their own cluster compared to other clusters.
                      Values range from -1 to 1, with higher values indicating better-defined clusters.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Loading overlay */}
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

export default KMeans;
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import './KMeans.css';

function KMeans() {
  const [data, setData] = useState({ X: [] });
  const [clusterCount, setClusterCount] = useState(3);
  const [maxIterations, setMaxIterations] = useState(100);
  const [sampleCount, setSampleCount] = useState(100);
  const [sampleClusters, setSampleClusters] = useState(3);
  const [clusterVariance, setClusterVariance] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [datasetType, setDatasetType] = useState('blobs');
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });
  const [canvasSize, setCanvasSize] = useState({ width: 400, height: 400 });
  const [canvasBounds, setCanvasBounds] = useState({ xMin: -10, xMax: 10, yMin: -10, yMax: 10 });
  const [dataPreview, setDataPreview] = useState(null);
  
  const canvasRef = useRef(null);
  const resultsRef = useRef(null);
  const apiUrl = 'http://127.0.0.1:5000';

  // Display notification
  const showNotification = (message, type = 'success') => {
    setNotification({ show: true, message, type });
    setTimeout(() => {
      setNotification({ show: false, message: '', type: '' });
    }, 3000);
  };

  // Generate sample data for clustering
  const generateSampleData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `${apiUrl}/api/kmeans/sample?n_samples=${sampleCount}&n_clusters=${sampleClusters}&variance=${clusterVariance}&dataset_type=${datasetType}`
      );
      setData(response.data);
      showNotification('Sample data generated successfully');
      generateDataPreview(response.data.X);
    } catch (error) {
      console.error('Error generating sample data:', error);
      showNotification('Failed to generate sample data', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Generate data preview
  const generateDataPreview = async (dataPoints) => {
    if (!dataPoints || dataPoints.length === 0) {
      // Clear canvas if no data points
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawEmptyCanvas();
      }
      return;
    }
    
    try {
      // Find data boundaries for better visualization
      const xValues = dataPoints.map(point => point[0]);
      const yValues = dataPoints.map(point => point[1]);
      
      // Calculate range
      const xRange = Math.max(...xValues) - Math.min(...xValues);
      const yRange = Math.max(...yValues) - Math.min(...yValues);
      
      // Add more generous padding (20% of range on each side)
      const xPadding = xRange * 0.2;
      const yPadding = yRange * 0.2;
      
      const xMin = Math.min(...xValues) - xPadding;
      const xMax = Math.max(...xValues) + xPadding;
      const yMin = Math.min(...yValues) - yPadding;
      const yMax = Math.max(...yValues) + yPadding;
      
      setCanvasBounds({ xMin, xMax, yMin, yMax });
      
      // Try using the canvas drawing directly instead of backend
      drawDataOnCanvas(dataPoints);
      setDataPreview(null);
    } catch (error) {
      console.error('Could not generate preview, using canvas fallback', error);
      // Fall back to canvas drawing
      drawDataOnCanvas(dataPoints);
    }
  };

  // Draw empty canvas with grid and axes only
  const drawEmptyCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const { xMin, xMax, yMin, yMax } = canvasBounds;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    drawGrid(ctx, width, height);
    
    // Draw axes
    drawAxes(ctx, width, height, xMin, xMax, yMin, yMax);
  };
  
  // Draw grid on canvas
  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    
    // Vertical grid lines
    const xStep = width / 10;
    for (let i = 0; i <= width; i += xStep) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    const yStep = height / 10;
    for (let i = 0; i <= height; i += yStep) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }
  };
  
  // Draw axes on canvas
  const drawAxes = (ctx, width, height, xMin, xMax, yMin, yMax) => {
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1.5;
    
    // X-axis
    const yZero = height - ((0 - yMin) / (yMax - yMin) * height);
    if (yZero >= 0 && yZero <= height) {
      ctx.beginPath();
      ctx.moveTo(0, yZero);
      ctx.lineTo(width, yZero);
      ctx.stroke();
      
      // Add labels for X-axis with more padding
      ctx.fillStyle = '#666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      const numLabels = 5;
      for (let i = 0; i <= numLabels; i++) {
        const x = width * (i / numLabels);
        const value = xMin + (i / numLabels) * (xMax - xMin);
        ctx.fillText(value.toFixed(1), x, yZero + 20); // Increased distance from axis
      }
    }
    
    // Y-axis
    const xZero = (0 - xMin) / (xMax - xMin) * width;
    if (xZero >= 0 && xZero <= width) {
      ctx.beginPath();
      ctx.moveTo(xZero, 0);
      ctx.lineTo(xZero, height);
      ctx.stroke();
      
      // Add labels for Y-axis with more padding
      ctx.fillStyle = '#666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      const numLabels = 5;
      for (let i = 0; i <= numLabels; i++) {
        const y = height * (i / numLabels);
        const value = yMax - (i / numLabels) * (yMax - yMin);
        ctx.fillText(value.toFixed(1), xZero - 10, y + 4); // Increased distance from axis
      }
    }
    
    // Add axis titles
    ctx.fillStyle = '#444';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    
    // X-axis title
    ctx.fillText('Feature 1', width / 2, height - 5);
    
    // Y-axis title (rotated)
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Feature 2', 0, 0);
    ctx.restore();
  };
  
  // Draw data points on canvas
  const drawDataOnCanvas = (dataPoints) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const { xMin, xMax, yMin, yMax } = canvasBounds;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = "#f9f9f9";
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid and axes
    drawGrid(ctx, width, height);
    drawAxes(ctx, width, height, xMin, xMax, yMin, yMax);
    
    // Draw points with gradient effect
    dataPoints.forEach((point, index) => {
      const x = ((point[0] - xMin) / (xMax - xMin)) * width;
      const y = height - ((point[1] - yMin) / (yMax - yMin)) * height;
      
      // Create a gradient fill
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
      gradient.addColorStop(0, '#5c6bc0'); // Inner color
      gradient.addColorStop(1, '#3f51b5'); // Outer color
      
      // Draw point with shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
      ctx.shadowBlur = 3;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
    });

    // Add subtle text indicator
    if (dataPoints.length === 0) {
      ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Click to add data points', width / 2, height / 2);
    }
  };

  // Run K-means clustering
  const runKMeans = async () => {
    try {
      if (!data || data.X.length < 3) {
        showNotification('Please generate or add at least 3 data points', 'error');
        return;
      }

      setLoading(true);
      const requestData = {
        ...data,
        k: clusterCount,
        max_iterations: maxIterations
      };

      const response = await axios.post(`${apiUrl}/api/kmeans`, requestData);
      setResults(response.data);
      showNotification('K-Means clustering completed successfully');
      
      // Scroll to results after a brief delay to allow rendering
      setTimeout(() => {
        if (resultsRef.current) {
          resultsRef.current.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
          });
        }
      }, 100);
    } catch (error) {
      console.error('Error running K-means clustering:', error);
      showNotification(error.response?.data?.error || 'Failed to run K-means clustering', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Clear all points
  const clearAllPoints = () => {
    setData({...data, X: []});
    drawEmptyCanvas(); // Immediately clear the canvas
    showNotification('All points cleared');
  };

  // Handle canvas click to add points
  const handleCanvasClick = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert canvas coordinates to data coordinates
    const { width, height } = canvas;
    const { xMin, xMax, yMin, yMax } = canvasBounds;
    
    const dataX = xMin + (x / width) * (xMax - xMin);
    const dataY = yMax - (y / height) * (yMax - yMin);
    
    // Add the new point to the dataset
    const newData = {
      ...data,
      X: [...(data.X || []), [dataX, dataY]]
    };
    
    setData(newData);
    drawDataOnCanvas(newData.X);
    showNotification('Point added', 'success');
  };

  // Set up canvas when component mounts
  useEffect(() => {
    const handleResize = () => {
      const container = document.querySelector('.preview-container');
      if (container) {
        // Make canvas larger to accommodate padding and labels
        const width = Math.min(container.clientWidth - 40, 520);
        setCanvasSize({
          width: width,
          height: width * 0.8 // Slightly reduced height for aspect ratio
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    handleResize();
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Redraw canvas when data or canvas size changes
  useEffect(() => {
    if (data.X && data.X.length > 0) {
      drawDataOnCanvas(data.X);
    } else {
      drawEmptyCanvas();
    }
  }, [data, canvasSize, canvasBounds]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="page-container"
    >
      {notification.show && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      <h1 className="page-title">K-Means Clustering</h1>
      <p className="description">
        K-means clustering is an unsupervised learning algorithm that partitions data into K clusters,
        where each data point belongs to the cluster with the nearest mean.
      </p>

      <div className="kmeans-layout">
        {/* Parameters Panel - Left Side */}
        <div className="parameters-panel card">
          <h2>Algorithm Parameters</h2>

          <div className="section-divider">
            <h3>Data Options</h3>
            
            <div className="form-group">
              <label>Dataset Type</label>
              <select 
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value)}
                className="select-input"
              >
                <option value="blobs">Isotropic Blobs</option>
                <option value="moons">Two Moons</option>
                <option value="circles">Concentric Circles</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>Number of Samples: {sampleCount}</label>
              <input
                type="range"
                min="50"
                max="500"
                step="50"
                value={sampleCount}
                onChange={(e) => setSampleCount(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div className="form-group">
              <label>True Clusters: {sampleClusters}</label>
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={sampleClusters}
                onChange={(e) => setSampleClusters(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div className="form-group">
              <label>Cluster Variance: {clusterVariance.toFixed(1)}</label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={clusterVariance}
                onChange={(e) => setClusterVariance(Number(e.target.value))}
                className="slider"
              />
            </div>

            <button
              className="button primary"
              onClick={generateSampleData}
              disabled={loading}
            >
              Generate Sample Data
            </button>
          </div>

          <div className="section-divider">
            <h3>K-Means Parameters</h3>

            <div className="form-group">
              <label>Number of Clusters (K): {clusterCount}</label>
              <input
                type="range"
                min="2"
                max="10"
                step="1"
                value={clusterCount}
                onChange={(e) => setClusterCount(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div className="form-group">
              <label>Max Iterations: {maxIterations}</label>
              <input
                type="range"
                min="10"
                max="300"
                step="10"
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                className="slider"
              />
            </div>

            <button
              className="button primary full-width"
              onClick={runKMeans}
              disabled={loading || !data || data.X.length < 3}
            >
              {loading ? 'Processing...' : 'Run K-Means Clustering'}
            </button>
          </div>
          
          {data && data.X && data.X.length > 0 && (
            <p className="data-info">
              {data.X.length} data points loaded
            </p>
          )}
        </div>
        
        {/* Data Preview Panel - Right Side */}
        <div className="preview-panel card">
          <div className="preview-header">
            <h2>Data Preview</h2>
            <div className="preview-header-content">
              <p className="help-text">
                Click on the plot to add data points manually.
              </p>
              
              <div className="button-group">
                <button 
                  className="button secondary" 
                  onClick={clearAllPoints}
                >
                  Clear All Points
                </button>
              </div>
            </div>
          </div>
          
          <div className="preview-container">
            <canvas
              ref={canvasRef}
              width={canvasSize.width}
              height={canvasSize.height}
              onClick={handleCanvasClick}
              className="data-canvas"
            ></canvas>
          </div>
          
          <div className="coordinates-info">
            <span>X Range: [{canvasBounds.xMin.toFixed(1)}, {canvasBounds.xMax.toFixed(1)}]</span>
            <span>Y Range: [{canvasBounds.yMin.toFixed(1)}, {canvasBounds.yMax.toFixed(1)}]</span>
          </div>
        </div>
      </div>

      {/* Results Area */}
      {results && (
        <div className="card results-card" ref={resultsRef}>
          <h2 className="results-title">Clustering Results</h2>
          
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Silhouette Score</h3>
              <div className="metric-value">
                {results.silhouette_score ? results.silhouette_score.toFixed(4) : 'N/A'}
              </div>
              <p className="metric-description">
                Range: [-1, 1] (Higher is better)
              </p>
            </div>

            <div className="metric-card">
              <h3>Iterations</h3>
              <div className="metric-value">
                {results.iterations || 'N/A'}
              </div>
              <p className="metric-description">
                Number of iterations until convergence
              </p>
            </div>

            <div className="metric-card">
              <h3>Inertia</h3>
              <div className="metric-value">
                {results.inertia ? results.inertia.toFixed(2) : 'N/A'}
              </div>
              <p className="metric-description">
                Sum of squared distances to nearest centroid
              </p>
            </div>
          </div>

          {/* Visualization */}
          {results.plot && (
            <div className="visualization">
              <img
                src={`data:image/png;base64,${results.plot}`}
                alt="K-Means Clustering Visualization"
                className="result-image"
              />
            </div>
          )}
          
          {/* Animation */}
          {results.animation && (
            <div className="visualization animation-container">
              <h3 className="animation-title">Clustering Animation</h3>
              <img
                src={`data:image/gif;base64,${results.animation}`}
                alt="K-Means Clustering Animation"
                className="result-image animation-image"
              />
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}

export default KMeans;
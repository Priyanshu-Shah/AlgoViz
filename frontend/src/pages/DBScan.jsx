import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import './ModelPage.css';

function DBScan() {
    const navigate = useNavigate();
    const [dataPoints, setDataPoints] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState("checking");
    const [algorithmStatus, setAlgorithmStatus] = useState("idle"); // "idle", "running", "completed"
    
    // Parameters
    const [eps, setEps] = useState(0.7);
    const [minPoints, setMinPoints] = useState(5);
    
    // Visualization options
    const [showCorePoints, setShowCorePoints] = useState(true);
    const [showBorderPoints, setShowBorderPoints] = useState(true);
    const [showNoisePoints, setShowNoisePoints] = useState(true);
    const [showEpsilonRadius, setShowEpsilonRadius] = useState(true);
    
    // Canvas refs and state
    const canvasRef = useRef(null);
    const canvasWidth = 600;
    const canvasHeight = 600;
    const scale = useMemo(() => ({
      x: { min: -8, max: 8 },
      y: { min: -8, max: 8 }
    }), []);
    
    // Algorithm data
    const [iterations, setIterations] = useState([]);
    const [currentIteration, setCurrentIteration] = useState(0);
    const [isAutoAdvancing, setIsAutoAdvancing] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(500); // milliseconds between iterations
    const [stats, setStats] = useState({
        processedPoints: 0,
        totalPoints: 0,
        numClusters: 0,
        numCorePoints: 0,
        numBorderPoints: 0,
        numNoisePoints: 0
    });
    
    // Sample data modal
    const [showSampleDataModal, setShowSampleDataModal] = useState(false);
    const [sampleCount, setSampleCount] = useState(100);
    const [sampleClusters, setSampleClusters] = useState(3);
    const [datasetType, setDatasetType] = useState('blobs');
    const [noiseLevel, setNoiseLevel] = useState(0.05);
    
    // Check backend health on component mount
    useEffect(() => {
        const checkBackendHealth = async () => {
            try {
                const response = await axios.get('http://localhost:5000/api/health');
                setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
            } catch (err) {
                console.error("Backend health check failed:", err);
                setBackendStatus("disconnected");
            }
        };
        
        checkBackendHealth();
    }, []);
    
    // Handle canvas click to add data points
    const handleCanvasClick = (e) => {
        if (algorithmStatus !== "idle") {
            return; // Don't allow adding points during algorithm execution
        }
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const rect = canvas.getBoundingClientRect();
        // Calculate click position relative to canvas
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        // Convert to data coordinates
        const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
        const dataY = scale.y.max - (y / canvas.height) * (scale.y.max - scale.y.min);
        
        const newPoint = { x: dataX, y: dataY };
        
        setDataPoints([...dataPoints, newPoint]);
    };
    
    // Sample data generation
    const loadSampleData = () => {
        setShowSampleDataModal(true);
    };
    
    const generateSampleData = async () => {
        try {
            setLoading(true);
            setError(null);
            
            const response = await axios.post('http://localhost:5000/api/dbscan/sample_data', {
                dataset_type: datasetType,
                n_samples: sampleCount,
                n_clusters: sampleClusters,
                noise_level: noiseLevel
            });
            
            if (response.data.points) {
                setDataPoints(response.data.points);
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
                        <option value="anisotropic">Anisotropic Blobs</option>
                        <option value="noisy_circles">Noisy Circles with Outliers</option>
                    </select>
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
                
                {/* Clusters (only for blobs) */}
                {datasetType === 'blobs' && (
                    <div style={{ marginBottom: '1.25rem' }}>
                        <label htmlFor="clusters-gen-slider" style={{ 
                            display: 'block', 
                            marginBottom: '0.5rem', 
                            fontWeight: '500', 
                            color: '#4b5563' 
                        }}>
                            Number of Clusters: {sampleClusters}
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
                )}
                
                {/* Noise level */}
                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="noise-slider" style={{ 
                        display: 'block', 
                        marginBottom: '0.5rem', 
                        fontWeight: '500', 
                        color: '#4b5563' 
                    }}>
                        Noise Level: {noiseLevel.toFixed(2)}
                    </label>
                    <input
                        id="noise-slider"
                        type="range"
                        min="0.01"
                        max="0.2"
                        step="0.01"
                        value={noiseLevel}
                        onChange={(e) => setNoiseLevel(Number(e.target.value))}
                        style={{ width: '100%' }}
                    />
                    <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        fontSize: '0.8rem', 
                        color: '#6b7280',
                        marginTop: '0.25rem'
                    }}>
                        <span>0.01 (Clean data)</span>
                        <span>0.20 (Noisy data)</span>
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
    
    // Reset data points
    const resetData = () => {
        setDataPoints([]);
        setIterations([]);
        setCurrentIteration(0);
        setAlgorithmStatus("idle");
        setIsAutoAdvancing(false);
        setStats({
            processedPoints: 0,
            totalPoints: 0,
            numClusters: 0,
            numCorePoints: 0,
            numBorderPoints: 0,
            numNoisePoints: 0
        });
        setError(null);
    };
    
    // Run the DBSCAN algorithm
    const runDBSCAN = async () => {
        try {
            setLoading(true);
            setError(null);
            setAlgorithmStatus("running");
            
            const response = await axios.post('http://localhost:5000/api/dbscan/run_complete', {
                points: dataPoints,
                eps: eps,
                min_samples: minPoints,
                show_core_points: showCorePoints,
                show_border_points: showBorderPoints,
                show_noise_points: showNoisePoints,
                show_epsilon_radius: showEpsilonRadius
            });
            
            if (response.data.status === "success") {
                setIterations(response.data.iterations);
                setCurrentIteration(0);
                setAlgorithmStatus("completed");
                
                // Update initial stats
                const iteration = response.data.iterations[0];
                setStats({
                    processedPoints: iteration.num_processed,
                    totalPoints: dataPoints.length,
                    numClusters: iteration.num_clusters,
                    numCorePoints: iteration.core_points.length,
                    numBorderPoints: iteration.border_points.length,
                    numNoisePoints: iteration.noise_points.length
                });
            } else {
                setError(`Error running DBSCAN: ${response.data.message || "Unknown error"}`);
                setAlgorithmStatus("idle");
            }
        } catch (err) {
            console.error("Error running DBSCAN:", err);
            setError(`Error running DBSCAN: ${err.message || "Unknown error"}`);
            setAlgorithmStatus("idle");
        } finally {
            setLoading(false);
        }
    };
    
    // Playback control methods
    const goToStart = () => {
        setCurrentIteration(0);
    };
    
    const goToEnd = () => {
        setCurrentIteration(iterations.length - 1);
    };
    
    const stepForward = () => {
        if (currentIteration < iterations.length - 1) {
            setCurrentIteration(prev => prev + 1);
        }
    };
    
    const stepBackward = () => {
        if (currentIteration > 0) {
            setCurrentIteration(prev => prev - 1);
        }
    };
    
    const goToIteration = (index) => {
        if (index >= 0 && index < iterations.length) {
            setCurrentIteration(index);
        }
    };
    
    const togglePlayback = () => {
        setIsAutoAdvancing(prev => !prev);
    };
    
    // Auto-advance through iterations
    useEffect(() => {
        let timer;
        
        if (isAutoAdvancing && algorithmStatus === "completed") {
            if (currentIteration < iterations.length - 1) {
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
    }, [currentIteration, iterations, isAutoAdvancing, playbackSpeed, algorithmStatus]);
    
    // Update stats when current iteration changes
    useEffect(() => {
        if (iterations.length > 0 && currentIteration < iterations.length) {
            const iteration = iterations[currentIteration];
            setStats({
                processedPoints: iteration.num_processed,
                totalPoints: dataPoints.length,
                numClusters: iteration.num_clusters,
                numCorePoints: iteration.core_points.length,
                numBorderPoints: iteration.border_points.length,
                numNoisePoints: iteration.noise_points.length
            });
        }
    }, [currentIteration, iterations, dataPoints.length]);
    
    // Canvas drawing effect
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
        
        // If no algorithm has been run yet, just draw the initial points
        if (algorithmStatus === "idle" || iterations.length === 0) {
            dataPoints.forEach(point => {
                const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
                const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;
                
                // Around line 550-560 (initial points section)
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                // Change this to gray instead of blue
                ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for unprocessed points
                ctx.fill();
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
            
            // Add subtle text indicator if no data
            if (dataPoints.length === 0) {
                ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
                ctx.font = '16px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Click to add data points', width / 2, height / 2);
            }
        } else if (iterations.length > 0 && currentIteration < iterations.length) {
            // Draw based on the current iteration
            const iteration = iterations[currentIteration];
            const labels = iteration.labels;
            
            // Draw epsilon radius for current point if requested
            if (showEpsilonRadius && iteration.current_point !== null) {
                const currentPoint = dataPoints[iteration.current_point];
                const x = ((currentPoint.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
                const y = ((scale.y.max - currentPoint.y) / (scale.y.max - scale.y.min)) * height;
                
                // Draw epsilon radius as dashed circle
                const epsPx = (eps / (scale.x.max - scale.x.min)) * width;
                ctx.beginPath();
                ctx.setLineDash([4, 4]);
                ctx.arc(x, y, epsPx, 0, Math.PI * 2);
                ctx.strokeStyle = '#3b82f6'; // Blue for epsilon radius
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.setLineDash([]);
            }
            
            // Draw all points based on their current state
            dataPoints.forEach((point, idx) => {
                const x = ((point.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
                const y = ((scale.y.max - point.y) / (scale.y.max - scale.y.min)) * height;
                
                const label = labels[idx];
                const isCurrentPoint = idx === iteration.current_point;
                const isCore = iteration.core_points.includes(idx);
                const isBorder = iteration.border_points.includes(idx);
                const isNoise = iteration.noise_points.includes(idx);
                
                // Draw the point
                ctx.beginPath();
                ctx.arc(x, y, isCurrentPoint ? 8 : 6, 0, Math.PI * 2);
                const isInitialIteration = currentIteration === 0 && iteration.phase === "initialization";
                // Fill color based on point type (matching KNN colors)
                if (isCurrentPoint) {
                  ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for processing point
              } else if (isInitialIteration) {
                  // If it's the initial iteration, keep all points gray
                  ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for unprocessed points
              } else if (isCore) {
                  // Get color based on cluster label
                  const clusterColors = [
                      'rgba(59, 130, 246, 0.7)', // Blue for cluster 0
                      'rgba(239, 68, 68, 0.7)', // Red for cluster 1
                      'rgba(34, 197, 94, 0.7)', // Green for cluster 2
                      'rgba(168, 85, 247, 0.7)', // Purple for cluster 3
                      'rgba(251, 146, 60, 0.7)' // Orange for cluster 4
                  ];
                  ctx.fillStyle = clusterColors[label % clusterColors.length];
              } else if (isBorder) {
                    if (label >= 0) {
                        // Get color based on cluster label
                        const clusterColors = [
                            'rgba(59, 130, 246, 0.7)', // Blue for cluster 0
                            'rgba(239, 68, 68, 0.7)', // Red for cluster 1
                            'rgba(34, 197, 94, 0.7)', // Green for cluster 2
                            'rgba(168, 85, 247, 0.7)', // Purple for cluster 3
                            'rgba(251, 146, 60, 0.7)' // Orange for cluster 4
                        ];
                        ctx.fillStyle = clusterColors[label % clusterColors.length];
                    } else {
                        ctx.fillStyle = 'rgba(251, 146, 60, 0.7)'; // Orange for border points not yet assigned
                    }
                } else if (isNoise && showNoisePoints) {
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for noise points
                } else if (label >= 0) {
                    // Get color based on cluster label
                    const clusterColors = [
                        'rgba(59, 130, 246, 0.7)', // Blue for cluster 0
                        'rgba(239, 68, 68, 0.7)', // Red for cluster 1
                        'rgba(34, 197, 94, 0.7)', // Green for cluster 2
                        'rgba(168, 85, 247, 0.7)', // Purple for cluster 3
                        'rgba(251, 146, 60, 0.7)' // Orange for cluster 4
                    ];
                    ctx.fillStyle = clusterColors[label % clusterColors.length];
                  } else if (label === -1) { // Noise
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for noise
                } else if (label === -2) { // Unprocessed
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for unprocessed
                } else if (label >= 0) { // Already assigned to a cluster
                    const clusterColors = [
                        'rgba(59, 130, 246, 0.7)', // Blue for cluster 0
                        'rgba(239, 68, 68, 0.7)', // Red for cluster 1
                        'rgba(34, 197, 94, 0.7)', // Green for cluster 2
                        'rgba(168, 85, 247, 0.7)', // Purple for cluster 3
                        'rgba(251, 146, 60, 0.7)' // Orange for cluster 4
                    ];
                    ctx.fillStyle = clusterColors[label % clusterColors.length];
                } else {
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray as fallback
                }
                
                ctx.fill();
                
                // Apply stroke style based on point type
                if (isCurrentPoint) {
                    ctx.strokeStyle = '#000'; // Black for current
                    ctx.lineWidth = 2;
                } else if (isCore && showCorePoints) {
                    ctx.strokeStyle = '#000'; // Black border for core
                    ctx.lineWidth = 2;
                } else if (isBorder && showBorderPoints) {
                    ctx.setLineDash([2, 2]); // Dotted for border
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = 1.5;
                } else if (isNoise && showNoisePoints) {
                    ctx.strokeStyle = '#6b7280'; // Gray for noise
                    ctx.lineWidth = 1;
                } else {
                    ctx.strokeStyle = '#333'; // Default border
                    ctx.lineWidth = 1;
                }
                
                ctx.stroke();
                ctx.setLineDash([]);
                
                // Draw connections to neighbors
                if (isCurrentPoint && showEpsilonRadius) {
                    iteration.neighbors.forEach(neighborIdx => {
                        if (neighborIdx !== idx) {
                            const neighbor = dataPoints[neighborIdx];
                            const nx = ((neighbor.x - scale.x.min) / (scale.x.max - scale.x.min)) * width;
                            const ny = ((scale.y.max - neighbor.y) / (scale.y.max - scale.y.min)) * height;
                            
                            // Draw connection line
                            ctx.beginPath();
                            ctx.setLineDash([2, 2]);
                            ctx.moveTo(x, y);
                            ctx.lineTo(nx, ny);
                            ctx.strokeStyle = '#3b82f6'; // Blue connection
                            ctx.lineWidth = 1;
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }
                    });
                }
            });
            
            // Add iteration info overlay
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.font = 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`Iteration: ${currentIteration + 1} / ${iterations.length}`, 10, 20);
            ctx.font = 'normal 12px Inter, sans-serif';
            ctx.fillText(`Phase: ${iteration.phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`, 10, 40);
        }
    }, [dataPoints, algorithmStatus, currentIteration, iterations, eps, scale, showCorePoints, showBorderPoints, showNoisePoints, showEpsilonRadius]);
    
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
                <h1 className="model-title">DBSCAN Clustering</h1>
            </div>

            <p className="model-description">
                Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a clustering algorithm that groups together points that are close to each other. It can find clusters of arbitrary shape and identify noise points.
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
                                    disabled={loading || dataPoints.length === 0}
                                    style={{
                                        padding: '0.5rem 0.75rem',
                                        backgroundColor: '#ef4444',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '0.375rem',
                                        cursor: 'pointer',
                                        fontSize: '0.875rem',
                                        fontWeight: '500',
                                        opacity: loading || dataPoints.length === 0 ? 0.7 : 1
                                    }}
                                >
                                    Reset
                                </button>
                            </div>
                        </div>
                        
                        <p style={{ marginBottom: '1rem', color: '#4b5563', fontSize: '0.875rem' }}>
                            {algorithmStatus === "idle" ? 
                                "Click on the canvas below to add data points. When ready, adjust the parameters and run DBSCAN." : 
                                "DBSCAN algorithm is running. Use the playback controls to see how clusters are formed step by step."}
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
                        
                        {algorithmStatus === "completed" && iterations.length > 0 && (
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
                                        disabled={currentIteration === iterations.length - 1}
                                        style={{
                                            padding: '0.5rem',
                                            backgroundColor: 'transparent',
                                            border: 'none',
                                            borderRadius: '0.25rem',
                                            cursor: 'pointer',
                                            color: currentIteration === iterations.length - 1 ? '#d1d5db' : '#374151'
                                        }}
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: 'rotate(180deg)' }}>
                                            <polygon points="19 20 9 12 19 4 19 20"></polygon>
                                        </svg>
                                    </button>
                                    
                                    <button
                                        onClick={goToEnd}
                                        disabled={currentIteration === iterations.length - 1}
                                        style={{
                                            padding: '0.5rem',
                                            backgroundColor: 'transparent',
                                            border: 'none',
                                            borderRadius: '0.25rem',
                                            cursor: 'pointer',
                                            color: currentIteration === iterations.length - 1 ? '#d1d5db' : '#374151'
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
                                            const newIteration = Math.floor(percentage * iterations.length);
                                            goToIteration(newIteration);
                                        }}>
                                            <div 
                                                style={{ 
                                                    position: 'absolute',
                                                    top: '0',
                                                    left: '0',
                                                    width: `${(currentIteration / Math.max(1, iterations.length - 1)) * 100}%`,
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
                                            left: `${(currentIteration / Math.max(1, iterations.length - 1)) * 100}%`,
                                            transform: 'translateX(-50%)',
                                            backgroundColor: '#3b82f6',
                                            color: 'white',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            fontSize: '0.7rem',
                                            fontWeight: '500'
                                        }}>
                                            {currentIteration + 1} of {iterations.length}
                                        </span>
                                    </div>
                                    
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                        <span style={{ fontSize: '0.8rem', color: '#6b7280', whiteSpace: 'nowrap' }}>Speed:</span>
                                        <input 
                                            type="range"
                                            min="5"
                                            max="4000"
                                            step="50"
                                            value={4050 - playbackSpeed}
                                            onChange={(e) => setPlaybackSpeed(4050- parseInt(e.target.value))}
                                            style={{ flex: 1 }}
                                        />
                                        <span style={{ fontSize: '0.8rem', color: '#374151' }}>
                                        {playbackSpeed < 500 ? 'Fast' : 'Slow'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}
                        
                        {algorithmStatus !== "idle" && (
                            <div className="stats-section" style={{
                                backgroundColor: 'white',
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                border: '1px solid #e5e7eb',
                                marginTop: '1rem'
                            }}>
                                <h3 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.75rem' }}>
                                    Clustering Statistics
                                </h3>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '0.75rem' }}>
                                    <p><strong>Points processed:</strong> {stats.processedPoints} / {stats.totalPoints}</p>
                                    <p><strong>Clusters found:</strong> {stats.numClusters}</p>
                                    <p><strong>Core points:</strong> {stats.numCorePoints}</p>
                                    <p><strong>Border points:</strong> {stats.numBorderPoints}</p>
                                    <p><strong>Noise points:</strong> {stats.numNoisePoints}</p>
                                    <p><strong>Step:</strong> {currentIteration + 1} / {iterations.length}</p>
                                </div>
                                
                                <div style={{
                                    width: '100%',
                                    height: '8px',
                                    backgroundColor: '#e5e7eb',
                                    borderRadius: '4px',
                                    overflow: 'hidden'
                                }}>
                                    <div 
                                        className="progress-fill" 
                                        style={{ 
                                            width: `${(currentIteration / Math.max(1, iterations.length - 1)) * 100}%`,
                                            height: '100%',
                                            backgroundColor: '#3b82f6',
                                            borderRadius: '4px',
                                            transition: 'width 0.3s ease'
                                        }}
                                    ></div>
                                </div>
                                
                                <p style={{ marginTop: '0.75rem', fontSize: '0.9rem', fontWeight: '500', color: '#4b5563' }}>
                                    Current phase: {iterations[currentIteration]?.phase?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Initializing'}
                                </p>
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
                            
                            <div style={{ marginBottom: '1.5rem' }}>
                                <label htmlFor="eps-param" style={{ 
                                    display: 'block', 
                                    marginBottom: '0.5rem', 
                                    fontWeight: '500', 
                                    color: '#4b5563' 
                                }}>
                                    Epsilon (radius): {eps}
                                </label>
                                <input 
                                    id="eps-param"
                                    type="range" 
                                    min="0.1" 
                                    max="2.0" 
                                    step="0.1"
                                    value={eps}
                                    onChange={(e) => setEps(parseFloat(e.target.value))}
                                    disabled={algorithmStatus !== "idle"}
                                    style={{ width: '100%' }}
                                />
                                <div style={{ 
                                    display: 'flex', 
                                    justifyContent: 'space-between', 
                                    fontSize: '0.85rem', 
                                    color: '#6b7280',
                                    marginTop: '0.5rem'
                                }}>
                                    <span>0.1 (Small neighborhood)</span>
                                    <span>2.0 (Large neighborhood)</span>
                                </div>
                            </div>
                            
                            <div style={{ marginBottom: '1.5rem' }}>
                                <label htmlFor="min-points-param" style={{ 
                                    display: 'block', 
                                    marginBottom: '0.5rem', 
                                    fontWeight: '500', 
                                    color: '#4b5563' 
                                }}>
                                    Minimum Points: {minPoints}
                                </label>
                                <input 
                                    id="min-points-param"
                                    type="range" 
                                    min="2" 
                                    max="15" 
                                    step="1"
                                    value={minPoints}
                                    onChange={(e) => setMinPoints(parseInt(e.target.value))}
                                    disabled={algorithmStatus !== "idle"}
                                    style={{ width: '100%' }}
                                />
                                <div style={{ 
                                    display: 'flex', 
                                    justifyContent: 'space-between', 
                                    fontSize: '0.85rem', 
                                    color: '#6b7280',
                                    marginTop: '0.5rem'
                                }}>
                                    <span>2 (Few neighbors needed)</span>
                                    <span>15 (Many neighbors needed)</span>
                                </div>
                            </div>
                            
                            <div style={{ marginBottom: '1.5rem' }}>
                                <h4 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500', color: '#4b5563' }}>
                                    Visualization Options
                                </h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                        <input 
                                            type="checkbox" 
                                            checked={showCorePoints}
                                            onChange={(e) => setShowCorePoints(e.target.checked)}
                                        />
                                        <span style={{ fontSize: '0.9rem' }}>Show Core Points</span>
                                    </label>
                                    
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                        <input 
                                            type="checkbox" 
                                            checked={showBorderPoints}
                                            onChange={(e) => setShowBorderPoints(e.target.checked)}
                                        />
                                        <span style={{ fontSize: '0.9rem' }}>Show Border Points</span>
                                    </label>
                                    
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                        <input 
                                            type="checkbox" 
                                            checked={showNoisePoints}
                                            onChange={(e) => setShowNoisePoints(e.target.checked)}
                                        />
                                        <span style={{ fontSize: '0.9rem' }}>Show Noise Points</span>
                                    </label>
                                    
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                        <input 
                                            type="checkbox" 
                                            checked={showEpsilonRadius}
                                            onChange={(e) => setShowEpsilonRadius(e.target.checked)}
                                        />
                                        <span style={{ fontSize: '0.9rem' }}>Show Epsilon Radius</span>
                                    </label>
                                </div>
                            </div>
                            
                            <div style={{ marginTop: '2rem' }}>
                                <button
                                    onClick={runDBSCAN}
                                    disabled={loading || dataPoints.length < 2 || algorithmStatus !== "idle"}
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
                                        opacity: (loading || dataPoints.length < 2 || algorithmStatus !== "idle") ? 0.7 : 1
                                    }}
                                >
                                    {loading ? 'Running...' : 'Run DBSCAN'}
                                </button>
                            </div>
                        </div>
                        
                        <div style={{ 
                            backgroundColor: 'white', 
                            padding: '1.5rem', 
                            borderRadius: '6px', 
                            border: '1px solid #e5e7eb',
                            width: '100%'
                        }}>
                            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>How DBSCAN Works</h3>
                            
                            <div style={{ fontSize: '0.9rem', color: '#4b5563', lineHeight: '1.5' }}>
                                <p style={{ marginBottom: '0.75rem' }}>
                                    <strong>DBSCAN</strong> (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together based on two parameters:
                                </p>
                                
                                <ol style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
                                    <li><strong>Epsilon (ε):</strong> The radius that defines a neighborhood around each point.</li>
                                    <li><strong>MinPoints:</strong> The minimum number of points required to form a dense region.</li>
                                </ol>
                                
                                <p style={{ marginBottom: '0.75rem' }}>
                                    Points are classified as:
                                </p>
                                
                                <ul style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>
                                    <li><strong>Core points:</strong> Have at least MinPoints neighbors within ε.</li>
                                    <li><strong>Border points:</strong> Are neighbors of core points but aren't core points themselves.</li>
                                    <li><strong>Noise points:</strong> Neither core nor border points.</li>
                                </ul>
                                
                                <p style={{ marginBottom: '0.75rem' }}>
                                    The algorithm works in two phases:
                                </p>
                                
                                <ol style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>
                                    <li><strong>Find core points:</strong> Identify all points that have at least MinPoints neighbors within ε distance.</li>
                                    <li><strong>Expand clusters:</strong> Starting from each core point, expand clusters by recursively adding all density-reachable points.</li>
                                </ol>
                                
                                <p>
                                    This algorithm can find arbitrarily shaped clusters and automatically identifies noise points.
                                </p>
                            </div>
                            
                            {/* Legend */}
                            <h4 style={{ marginTop: '1.25rem', marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500' }}>Legend</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '1px solid #333' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Initial Points</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '1px solid #000' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Current Point</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.7)', border: '2px solid #000' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Core Point</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(239, 68, 68, 0.7)', border: '2px dashed #333' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Border Point</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'rgba(156, 163, 175, 0.7)', border: '1px solid #6b7280' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Noise Point</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', border: '2px dashed #3b82f6', backgroundColor: 'transparent' }}></div>
                                    <span style={{ fontSize: '0.85rem' }}>Epsilon Radius</span>
                                </div>
                            </div>
                        </div>
                        
                        {/* Debugging info (visible during development) */}
                        {iterations.length > 0 && iterations[currentIteration] && (
                            <div style={{ 
                                marginTop: '1.5rem',
                                backgroundColor: '#f8fafc', 
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                border: '1px solid #e5e7eb'
                            }}>
                                <details>
                                    <summary style={{ fontWeight: '500', cursor: 'pointer', marginBottom: '0.5rem' }}>Debug Information</summary>
                                    <div style={{ fontSize: '0.85rem', color: '#4b5563' }}>    
                                        <p><strong>Phase:</strong> {iterations[currentIteration].phase}</p>
                                        <p><strong>Current Point:</strong> {iterations[currentIteration].current_point !== null ? `Index ${iterations[currentIteration].current_point}` : 'None'}</p>
                                        <p><strong>Neighbors:</strong> {iterations[currentIteration].neighbors ? 
                                            iterations[currentIteration].neighbors.join(', ') : 'None'}</p>
                                        <p><strong>Message:</strong> {iterations[currentIteration].message}</p>
                                    </div>
                                </details>
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
                    <style jsx>{`
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

export default DBScan;
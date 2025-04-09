/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { runSVM, getSVMSampleData, checkHealth } from '../api';

function SVM(){
    const navigate = useNavigate();
    const [dataPairs, setDataPairs] = useState([{ x: '', y: '' }]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState("checking");
    const [sampleLoading, setSampleLoading] = useState(false);
    const [currentClass, setCurrentClass] = useState(0);
    const [kernel, setKernel] = useState('linear');
    const [decisionBoundary, setDecisionBoundary] = useState(null);
    const [supportVectors, setSupportVectors] = useState([]);
    const [interactionMode, setInteractionMode] = useState('train');
    const [predictedPoints, setPredictedPoints] = useState([]);


    // Canvas ref and state for interactive plotting
    const canvasRef = useRef(null);
    const [canvasDimensions] = useState({ width: 600, height: 600 }); // Square canvas
    // Fixed scale - adjust to -8 to 8 range
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

    // Helper function to draw grid and axes
    const drawGrid = (ctx, canvas) => {
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 0.5;
        
        // Draw horizontal grid lines
        for (let i = 0; i <= 16; i++) {
        const y = canvas.height - (i / 16) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
        }
        
        // Draw vertical grid lines
        for (let i = 0; i <= 16; i++) {
        const x = (i / 16) * canvas.width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
        }
        
        // Draw axes with dotted lines
        ctx.strokeStyle = '#888';
        ctx.lineWidth = 1.5;
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
        ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
        
        // X-axis labels
        for (let i = 0; i <= 16; i += 2) {
        const x = (i / 16) * canvas.width;
        const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
        ctx.fillText(value.toFixed(0), x - 10, canvas.height - 5);
        }
        
        // Y-axis labels
        for (let i = 0; i <= 16; i += 2) {
        const y = canvas.height - (i / 16) * canvas.height;
        const value = scale.y.min + (i / 16) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(0), 5, y + 4);
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
    
            if (point.isPredicted) {
                // Predicted points: Use a dotted boundary
                ctx.fillStyle = point.predictedClass === 0 ? 'rgba(59, 130, 246, 0.5)' : 'rgba(239, 68, 68, 0.5)';
                ctx.setLineDash([4, 4]); // Dotted line
            } else {
                // Regular points
                ctx.fillStyle = point.class === 0 ? 'rgba(59, 130, 246, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                ctx.setLineDash([]); // Solid line
            }
    
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    };
    
    // Convert screen coordinates to data coordinates
    const screenToData = (x, y) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        
        const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
        const dataY = scale.y.min + ((canvas.height - y) / canvas.height) * (scale.y.max - scale.y.min);
        
        return { x: dataX, y: dataY };
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw the grid and points
        drawGrid(ctx, canvas);
        drawPoints(ctx, canvas, [...getValidPoints(), ...predictedPoints]);
        
        // If a decision boundary image is available, you can also draw it as a background overlay:
        if (decisionBoundary) {
          const img = new Image();
          img.onload = () => {
              // Draw the decision boundary image covering the entire canvas
              ctx.globalAlpha = 0.5; // Adjust transparency as needed
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
              ctx.globalAlpha = 1.0; // Reset alpha
              // Then draw the support vector markers on top
              drawSupportVectorLines(ctx, canvas);
          };
          img.src = `data:image/png;base64,${decisionBoundary}`;
        } else {
          // If no decision boundary image, just draw support vector markers
          drawSupportVectorLines(ctx, canvas);
        }
    }, [dataPairs, predictedPoints, decisionBoundary, supportVectors]);

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
        const newPoint = {
            x: dataPoint.x.toFixed(2),
            y: dataPoint.y.toFixed(2),
            class: currentClass // Assign the selected class
        };
    
        setDataPairs([...dataPairs, newPoint]);
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
            const sampleData = await getSVMSampleData();
            
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

    const getValidPoints = () => {
        return dataPairs
            .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
            .map(pair => ({
                x: parseFloat(pair.x),
                y: parseFloat(pair.y),
                class: pair.class // Include class
            }));
    };

    // Helper function to draw support vector markers (as "X" markers)
    const drawSupportVectorLines = (ctx, canvas) => {
        if (!supportVectors || supportVectors.length === 0) return;
        supportVectors.forEach(sv => {
            // sv is assumed to be an array [x, y] in feature space
            const x = ((sv[0] - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
            const y = canvas.height - ((sv[1] - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
            ctx.beginPath();
            ctx.setLineDash([4, 4]); // Dashed line pattern
            const offset = 8; // Marker size
            // Draw diagonal from top-left to bottom-right
            ctx.moveTo(x - offset, y - offset);
            ctx.lineTo(x + offset, y + offset);
            // Draw diagonal from top-right to bottom-left
            ctx.moveTo(x + offset, y - offset);
            ctx.lineTo(x - offset, y + offset);
            ctx.strokeStyle = '#FF5733'; // Distinct color for support vectors
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
        });
    };

    const handleRunModel = async () => {
        // Filter out empty pairs
        const validPairs = getValidPoints();
    
        // Validate inputs
        if (validPairs.length < 2) {
            setError('Please add at least 2 data points for meaningful SVM results.');
            return;
        }
    
        setError(null);
        setLoading(true);
        setResults(null); // Reset results before new request
    
        try {
            // Prepare data for the API
            const apiData = {
                X: validPairs.map(pair => ({ x: pair.x, y: pair.y })), // Include x and y
                y: validPairs.map(pair => pair.class), // Include class labels
                kernel: kernel
            };
    
            console.log('Calling API endpoint with data:', apiData);
            const response = await runSVM(apiData);
    
            if (response.error) {
                throw new Error(response.error);
            }
    
            console.log('Results state set to:', response);
            setResults(response);
            setDecisionBoundary(response.decisionBoundary);
            setSupportVectors(response.supportVectors);
        } catch (err) {
            console.error('Error details:', err);
            setError(`Error: ${err.message || 'An error occurred while running the model.'}`);
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = () => {
        if (!results || !results.decisionBoundary) {
            setError('Please run the SVM model first to generate a decision boundary.');
            return;
        }
    
        const validPairs = getValidPoints();
    
        // Validate inputs
        if (validPairs.length === 0) {
            setError('Please add at least one point to predict.');
            return;
        }
    
        setError(null);
    
        // Use the decision boundary to classify the points
        const newPredictedPoints = validPairs.map(point => {
            // Simulate classification using the decision boundary
            const predictedClass = classifyPointWithBoundary(point, results.decisionBoundary);
    
            return {
                ...point,
                predictedClass, // Assign the predicted class
                isPredicted: true // Mark as a predicted point
            };
        });
    
        setPredictedPoints(newPredictedPoints);
    };

    // Helper function to classify a point using the decision boundary
    const classifyPointWithBoundary = (point, decisionBoundary) => {
        // Example logic: Use the decision boundary to classify the point
        // This is a placeholder. Replace with actual logic based on your decision boundary format.
        const { x, y } = point;
        if (decisionBoundary) {
            // Example: Simple threshold-based classification
            return x + y > 0 ? 1 : 0; // Replace with actual decision boundary logic
        }
        return 0; // Default class
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
                <h1 className="model-title">Support Vector Machine</h1>
            </div>
    
            <p className="model-description">
                Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks. 
                It finds the optimal hyperplane to separate data points into different classes.
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
                gridTemplateColumns: '2fr 1fr', 
                gap: '2rem',
                alignItems: 'start'
            }}>
                {/* Left Column: Interactive Canvas */}
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
                </div>

                {/* Right Column: Controls and Results */}
                <div>
                    <h2 style={{ marginBottom: '1rem', fontSize: '1.5rem', fontWeight: '600', color: '#1f2937' }}>
                        Controls and Results
                    </h2>

                    {/* Control Box */}
                    <div style={{ 
                        marginBottom: '1.5rem', 
                        backgroundColor: 'white', 
                        padding: '1.5rem', 
                        borderRadius: '6px', 
                        border: '1px solid #e5e7eb',
                        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                    }}>
                        <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem', fontWeight: '500' }}>
                            Controls
                        </h3>

                        {/* Kernel Selector */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <label htmlFor="kernel-selector" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>
                                Kernel Type
                            </label>
                            <select
                                id="kernel-selector"
                                value={kernel}
                                onChange={(e) => setKernel(e.target.value)}
                                style={{
                                    width: '100%',
                                    padding: '10px',
                                    border: '1px solid #d1d5db',
                                    borderRadius: '6px',
                                    backgroundColor: 'white',
                                    fontSize: '1rem',
                                    color: '#4b5563',
                                    cursor: 'pointer'
                                }}
                            >
                                <option value="linear">Linear</option>
                                <option value="poly">Polynomial</option>
                                <option value="rbf">RBF (Radial Basis Function)</option>
                                <option value="sigmoid">Sigmoid</option>
                            </select>
                        </div>

                        {/* Class Switch */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <h4 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500', color: '#4b5563' }}>
                                Select Class
                            </h4>
                            <div style={{ display: 'flex', gap: '0.75rem' }}>
                                <button
                                    onClick={() => setCurrentClass(0)}
                                    style={{
                                        padding: '0.75rem 1rem',
                                        backgroundColor: currentClass === 0 ? '#3b82f6' : '#e5e7eb',
                                        color: currentClass === 0 ? 'white' : '#4b5563',
                                        border: 'none',
                                        borderRadius: '0.5rem',
                                        cursor: 'pointer',
                                        fontWeight: '500',
                                        flex: 1,
                                        fontSize: '0.95rem'
                                    }}
                                >
                                    Class 0
                                </button>
                                <button
                                    onClick={() => setCurrentClass(1)}
                                    style={{
                                        padding: '0.75rem 1rem',
                                        backgroundColor: currentClass === 1 ? '#ef4444' : '#e5e7eb',
                                        color: currentClass === 1 ? 'white' : '#4b5563',
                                        border: 'none',
                                        borderRadius: '0.5rem',
                                        cursor: 'pointer',
                                        fontWeight: '500',
                                        flex: 1,
                                        fontSize: '0.95rem'
                                    }}
                                >
                                    Class 1
                                </button>
                            </div>
                        </div>

                        {/* Interaction Mode */}
                        <div>
                            <h4 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500', color: '#4b5563' }}>
                                Interaction Mode
                            </h4>
                            <div style={{ display: 'flex', gap: '0.75rem' }}>
                                <button
                                    onClick={() => setInteractionMode('train')}
                                    style={{
                                        padding: '0.75rem 1rem',
                                        backgroundColor: interactionMode === 'train' ? '#3b82f6' : '#e5e7eb',
                                        color: interactionMode === 'train' ? 'white' : '#4b5563',
                                        border: 'none',
                                        borderRadius: '0.5rem',
                                        cursor: 'pointer',
                                        fontWeight: '500',
                                        flex: 1,
                                        fontSize: '0.95rem'
                                    }}
                                >
                                    Training
                                </button>
                                <button
                                    onClick={() => setInteractionMode('test')}
                                    style={{
                                        padding: '0.75rem 1rem',
                                        backgroundColor: interactionMode === 'test' ? '#3b82f6' : '#e5e7eb',
                                        color: interactionMode === 'test' ? 'white' : '#4b5563',
                                        border: 'none',
                                        borderRadius: '0.5rem',
                                        cursor: 'pointer',
                                        fontWeight: '500',
                                        flex: 1,
                                        fontSize: '0.95rem'
                                    }}
                                >
                                    Testing
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Result Box */}
                    <div style={{ 
                        marginBottom: '1.5rem', 
                        backgroundColor: 'white', 
                        padding: '1.5rem', 
                        borderRadius: '6px', 
                        border: '1px solid #e5e7eb',
                        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                    }}>
                        <h3 style={{ marginBottom: '1rem', fontSize: '1.2rem', fontWeight: '500' }}>
                            Actions
                        </h3>

                        {/* Run SVM Button */}
                        <button 
                            onClick={handleRunModel}
                            disabled={loading || backendStatus === "disconnected"}
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
                                opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1,
                                marginBottom: '1rem'
                            }}
                        >
                            {loading ? 'Running...' : 'Run SVM'}
                        </button>

                        {/* Predict Button */}
                        <button 
                            onClick={handlePredict}
                            disabled={loading || backendStatus === "disconnected"}
                            style={{
                                width: '100%',
                                backgroundColor: loading ? '#93c5fd' : '#10b981',
                                color: 'white',
                                padding: '0.9rem',
                                fontSize: '1.05rem',
                                fontWeight: '500',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: loading ? 'wait' : 'pointer',
                                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                                opacity: (loading || backendStatus === "disconnected") ? 0.7 : 1
                            }}
                        >
                            {loading ? 'Predicting...' : 'Predict Testing Points'}
                        </button>
                    </div>

                    <div style={{ marginTop: '1.5rem' }}>
                    {loading ? (
                        <p>Loading...</p>
                    ) : results ? (
                        <div>
                        {/* Accuracy Box */}
                        <div style={{
                            backgroundColor: '#f3f4f6',
                            padding: '1rem',
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb',
                            marginBottom: '1rem',
                            textAlign: 'center'
                        }}>
                            <p style={{
                                fontSize: '1.2rem',
                                fontWeight: '600',
                                color: '#111827'
                            }}>
                            Accuracy: {results.accuracy}
                            </p>
                        </div>
                        {/* Decision Boundary Image */}
                        {decisionBoundary && (
                            <div style={{
                            textAlign: 'center'
                            }}>
                            <h4 style={{ marginBottom: '0.5rem' }}>Decision Boundary:</h4>
                            <img 
                                src={`data:image/png;base64,${decisionBoundary}`} 
                                alt="Decision Boundary"
                                style={{ width: '100%', maxWidth: '400px' }}
                            />
                            </div>
                        )}
                        </div>
                    ) : (
                        <p>No results yet. Run the model to see results.</p>
                    )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default SVM;
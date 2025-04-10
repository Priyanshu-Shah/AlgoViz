/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios'; // Added axios import
import { runSVM, getSVMSampleData, checkHealth } from '../api';

function SVM(){
    const navigate = useNavigate();
    const [dataPairs, setDataPairs] = useState([{ x: '', y: '', class: 0 }]);
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
    const [showSupportVectors, setShowSupportVectors] = useState(true);
    
    // Sample data generation state variables
    const [showSampleDataModal, setShowSampleDataModal] = useState(false);
    const [sampleDataType, setSampleDataType] = useState('blobs');
    const [sampleCount, setSampleCount] = useState(40);
    const [sampleClusters, setSampleClusters] = useState(2);
    const [sampleVariance, setSampleVariance] = useState(0.5);
    
    const [hyperparams, setHyperparams] = useState({
        gamma: 0.1,
        degree: 3,
        coef0: 0.0,
        marginWidth: 1.0
    });

    // Canvas ref and state for interactive plotting
    const canvasRef = useRef(null);
    const tempCanvasRef = useRef(null);
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

    // Helper function to update hyperparameters based on kernel selection
    useEffect(() => {
        // Reset relevant hyperparameters when kernel changes
        if (kernel === 'linear') {
            setHyperparams(prev => ({
                ...prev,
                gamma: 0.1,
                degree: 3,
                coef0: 0.0
            }));
        } else if (kernel === 'poly') {
            setHyperparams(prev => ({
                ...prev,
                gamma: 0.1,
                degree: 3,
                coef0: 1.0
            }));
        } else if (kernel === 'rbf') {
            setHyperparams(prev => ({
                ...prev,
                gamma: 0.1,
                degree: 3,
                coef0: 0.0
            }));
        } else if (kernel === 'sigmoid') {
            setHyperparams(prev => ({
                ...prev,
                gamma: 0.1,
                degree: 3,
                coef0: 1.0
            }));
        }
        
        // Reset predictions when kernel changes
        if (predictedPoints.length > 0) {
            setPredictedPoints([]);
        }
    }, [kernel]);

    // Reset predictions when any hyperparameter changes
    useEffect(() => {
        if (predictedPoints.length > 0 && results) {
            setPredictedPoints([]);
        }
    }, [hyperparams.gamma, hyperparams.degree, hyperparams.coef0, hyperparams.marginWidth]);

    // Update drawGrid function
    const drawGrid = (ctx, canvas) => {
        // Calculate step size for grid lines (16 divisions)
        const stepX = canvas.width / 16;
        const stepY = canvas.height / 16;
        
        // Draw grid
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 0.5;
        
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
        
        // Draw X and Y axes with dotted lines
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
            const value = scale.x.min + (i / 16) * (scale.x.max - scale.x.min);
            ctx.fillText(value.toFixed(0), x - 8, canvas.height - 5);
        }
        
        // Y-axis labels
        for (let i = 0; i <= 16; i += 2) {
            const y = i * stepY;
            const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
            ctx.fillText(value.toFixed(0), 5, y + 4);
        }
    };

    // Helper function to identify if a point is a support vector
    const isSupportVector = (point) => {
        if (!supportVectors) return false;
        
        // Check if the point coordinates match any support vector
        for (const sv of supportVectors) {
            if (Math.abs(parseFloat(point.x) - sv[0]) < 0.01 && 
                Math.abs(parseFloat(point.y) - sv[1]) < 0.01) {
                return true;
            }
        }
        return false;
    };

    // Update the drawPoints function
    const drawPoints = (ctx, canvas, points) => {
        if (!points || points.length === 0) return;
        
        // First draw regular points
        points.forEach(point => {
            // Skip drawing if this point is a support vector and showSupportVectors is true
            if (showSupportVectors && supportVectors && isSupportVector(point)) {
                return;
            }
            
            // Convert data coordinates to screen coordinates
            const x = ((parseFloat(point.x) - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
            const y = canvas.height - ((parseFloat(point.y) - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;

            if (point.isPredicted) {
                // Draw predicted points as squares
                const squareSize = 10; // Size of the square
                ctx.beginPath();
                ctx.rect(x - squareSize/2, y - squareSize/2, squareSize, squareSize);
                
                // Predicted points that haven't been classified yet are gray
                if (point.predictedClass === undefined) {
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for unpredicted points
                } else {
                    // Use class colors for predicted points
                    ctx.fillStyle = point.predictedClass === 0 ? 
                        'rgba(59, 130, 246, 0.7)' : // Blue for class 0
                        'rgba(239, 68, 68, 0.7)';  // Red for class 1
                }
                
                // Dotted border for prediction points
                ctx.setLineDash([2, 2]);
            } else {
                // Regular training points as circles
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                
                // Regular training points
                ctx.fillStyle = point.class === 0 ? 
                    'rgba(59, 130, 246, 0.7)' : // Blue for class 0
                    'rgba(239, 68, 68, 0.7)';  // Red for class 1
                ctx.setLineDash([]); // Solid line
            }

            ctx.fill();
            ctx.strokeStyle = '#000000'; // Black for outline
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
        });
    };

    // Update support vector markers drawing function
    const drawSupportVectorMarkers = (ctx, canvas) => {
        if (!supportVectors || supportVectors.length === 0) return;
        
        supportVectors.forEach((sv, index) => {
            // Convert data coordinates to screen coordinates
            const screenCoords = dataToScreen(sv[0], sv[1]);
            const x = screenCoords.x;
            const y = screenCoords.y;
            
            // Draw thicker circle around support vector
            ctx.beginPath();
            ctx.arc(x, y, 6,0, Math.PI * 2);
            ctx.strokeStyle = '#000000'; // Black for support vector outline
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Fill with class color but darker to highlight
            const svClass = results?.supportVectorClasses?.[index] ?? 0;
            ctx.fillStyle = svClass === 0 ? 
                'rgba(30, 80, 200, 0.85)' : // Darker blue for class 0
                'rgba(200, 30, 30, 0.85)';  // Darker red for class 1
            ctx.fill();
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

    // Convert data coordinates to screen coordinates
    const dataToScreen = (dataX, dataY) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        
        const x = ((dataX - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
        const y = canvas.height - ((dataY - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
        
        return { x, y };
    };

    // Update the useEffect for canvas drawing
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw background
        ctx.fillStyle = "#f9f9f9";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid and axes
        drawGrid(ctx, canvas);
        
        // If a decision boundary image is available, draw it first
        if (decisionBoundary) {
            const img = new Image();
            
            // Add error handling for image loading
            img.onerror = () => {
                console.error("Failed to load decision boundary image");
                setError("Failed to display decision boundary. The image may be corrupted.");
            };
            
            img.onload = () => {
                // Draw the decision boundary image covering the entire canvas
                ctx.globalAlpha = 0.7; // Make it semi-transparent
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.globalAlpha = 1.0; // Reset alpha
                
                // Then draw the points on top
                drawPoints(ctx, canvas, [...getValidPoints(), ...predictedPoints]);
                
                // Draw support vector markers if enabled
                if (showSupportVectors && supportVectors && supportVectors.length) {
                    drawSupportVectorMarkers(ctx, canvas);
                }
            };
            
            // Set the image source
            img.src = `data:image/png;base64,${decisionBoundary}`;
        } else {
            // Just draw points if no decision boundary
            drawPoints(ctx, canvas, [...getValidPoints(), ...predictedPoints]);
        }
    }, [dataPairs, predictedPoints, decisionBoundary, supportVectors, showSupportVectors]);

    // Update the handleCanvasClick function for better mapping
    const handleCanvasClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        // Calculate click position relative to canvas
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        const dataPoint = screenToData(x, y);
        console.log("Canvas clicked at data coordinates:", dataPoint);

        // If in training mode, add to dataPairs
        if (interactionMode === 'train') {
            const newPoint = {
                x: dataPoint.x.toFixed(2),
                y: dataPoint.y.toFixed(2),
                class: currentClass
            };
            setDataPairs([...dataPairs, newPoint]);
            
            // Reset any prediction results when adding training points
            setDecisionBoundary(null);
            setResults(null);
            setSupportVectors([]);
            setPredictedPoints([]);
        } else {
            // In testing mode, add to predicted points with gray color initially
            const pointToPredict = {
                x: parseFloat(dataPoint.x.toFixed(2)),
                y: parseFloat(dataPoint.y.toFixed(2)),
                isPredicted: true
                // Note: No predictedClass yet, will be added after prediction
            };
            
            setPredictedPoints([...predictedPoints, pointToPredict]);
        }
    };

    const handleAddPair = () => {
        setDataPairs([...dataPairs, { x: '', y: '', class: 0 }]);
    };

    const handleRemovePair = (index) => {
        const newPairs = [...dataPairs];
        newPairs.splice(index, 1);
        
        // Ensure there's always at least one row
        if (newPairs.length === 0) {
            newPairs.push({ x: '', y: '', class: 0 });
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

    // Update the loadSampleData function to show the modal
    const loadSampleData = () => {
        setShowSampleDataModal(true);
    };

    // Add a function to generate sample data with options
    const generateSampleDataWithOptions = () => {
        setLoading(true);
        setShowSampleDataModal(false);
        
        // Call the backend API
        axios.post(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/api/svm/sample_data`, {
            dataset_type: sampleDataType,
            count: sampleCount,
            n_clusters: sampleClusters,
            variance: sampleVariance
        })
        .then(response => {
            if (response.data && response.data.X && response.data.y) {
                // Convert sample data into points
                const pairs = [];
                for (let i = 0; i < response.data.X.length; i++) {
                    pairs.push({
                        x: parseFloat(response.data.X[i][0]).toFixed(2),
                        y: parseFloat(response.data.X[i][1]).toFixed(2),
                        class: parseInt(response.data.y[i])
                    });
                }
                
                setDataPairs(pairs);
                
                // Reset other states
                setPredictedPoints([]);
                setResults(null);
                setDecisionBoundary(null);
                setSupportVectors([]);
            }
        })
        .catch(err => {
            console.error("Error generating sample data:", err);
            setError("Failed to generate sample data: " + (err.response?.data?.error || err.message));
            
            // Fallback to standard sample data if advanced generation fails
            getSVMSampleData().then(sampleData => {
                if (!sampleData || !sampleData.X) {
                    setError("Could not load sample data");
                    return;
                }
                
                const pairs = [];
                for (let i = 0; i < sampleData.X.length; i++) {
                    pairs.push({
                        x: parseFloat(sampleData.X[i][0]).toFixed(2),
                        y: parseFloat(sampleData.X[i][1]).toFixed(2),
                        class: parseInt(sampleData.y[i])
                    });
                }
                setDataPairs(pairs);
            });
        })
        .finally(() => {
            setLoading(false);
        });
    };

    const resetData = () => {
        setDataPairs([{ x: '', y: '', class: 0 }]);
        setResults(null);
        setError(null);
        setPredictedPoints([]);
        setDecisionBoundary(null);
        setSupportVectors([]);
    };

    const getValidPoints = () => {
        return dataPairs
            .filter(pair => pair.x !== '' && pair.y !== '' && !isNaN(parseFloat(pair.x)) && !isNaN(parseFloat(pair.y)))
            .map(pair => ({
                x: parseFloat(pair.x),
                y: parseFloat(pair.y),
                class: pair.class
            }));
    };

    const handleRunModel = async () => {
        // Filter out empty pairs
        const validPairs = getValidPoints();
    
        // Validate inputs
        if (validPairs.length < 2) {
            setError('Please add at least 2 data points for meaningful SVM results.');
            return;
        }
    
        // Check that we have at least two classes
        const uniqueClasses = [...new Set(validPairs.map(p => p.class))];
        if (uniqueClasses.length < 2) {
            setError('Please add points from at least two different classes for classification.');
            return;
        }
    
        setError(null);
        setLoading(true);
        setPredictedPoints([]); // Reset predicted points
        setResults(null); // Reset results before new request
    
        try {
            // Prepare data for the API
            const apiData = {
                X: validPairs.map(pair => ({ x: parseFloat(pair.x), y: parseFloat(pair.y) })),
                y: validPairs.map(pair => parseInt(pair.class)),
                kernel: kernel,
                gamma: kernel === 'rbf' || kernel === 'poly' || kernel === 'sigmoid' ? 
                    (hyperparams.gamma === 'scale' || hyperparams.gamma === 'auto' ? 
                        hyperparams.gamma : parseFloat(hyperparams.gamma)) : 
                    'auto',
                degree: parseInt(hyperparams.degree),
                coef0: parseFloat(hyperparams.coef0),
                marginWidth: parseFloat(hyperparams.marginWidth)
            };
    
            console.log('Calling API endpoint with data:', apiData);
            const response = await runSVM(apiData);
    
            if (response.error) {
                throw new Error(response.error);
            }
    
            console.log('Results state set to:', response);
            
            // Handle response based on property names available
            setResults(response);
            
            // Handle the decision boundary image
            if (response.decisionBoundary) {
                setDecisionBoundary(response.decisionBoundary);
            } else if (response.decision_boundary) {
                setDecisionBoundary(response.decision_boundary);
            }
            
            // Handle support vectors
            if (response.supportVectors) {
                setSupportVectors(response.supportVectors);
            } else if (response.support_vectors) {
                setSupportVectors(response.support_vectors);
            }
        } catch (err) {
            console.error('Error details:', err);
            setError(`Error: ${err.message || 'An error occurred while running the model.'}`);
        } finally {
            setLoading(false);
        }
    };

    // Update the handlePredict function
    const handlePredict = () => {
        if (!results || !results.model_info) {
            setError('Please run the SVM model first to generate a decision boundary.');
            return;
        }

        if (predictedPoints.length === 0) {
            setError('Please add at least one point to predict by clicking on the canvas in testing mode.');
            return;
        }

        setError(null);
        setLoading(true);

        try {
            // Use the decision boundary to classify the points
            const newPredictedPoints = [...predictedPoints].map(point => {
                // Use the SVM model to classify the point
                const predictedClass = classifyPointWithSVMModel(point);

                return {
                    ...point,
                    predictedClass, // Assign the predicted class
                    isPredicted: true // Mark as a predicted point
                };
            });

            setPredictedPoints(newPredictedPoints);
        } catch (err) {
            setError(`Error during prediction: ${err.message}`);
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // Improved function to classify a point using SVM model
    const classifyPointWithSVMModel = (point) => {
        if (!results || !results.model_info) return 0;
        
        // If we have a decision boundary image, we can make predictions directly
        // by querying values from the image data
        if (decisionBoundary && canvasRef.current) {
            // Get point coordinates on canvas
            const { x, y } = dataToScreen(point.x, point.y);
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            
            // Set up a temporary canvas to check the color at the point's position
            if (tempCanvasRef.current) {
                const tempCanvas = tempCanvasRef.current;
                const tempCtx = tempCanvas.getContext('2d');
                const img = new Image();
                
                img.onload = () => {
                    // Draw the decision boundary image on temp canvas
                    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
                    tempCtx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);
                };
                
                img.src = `data:image/png;base64,${decisionBoundary}`;
                
                // This is a synchronous approach since we need the result immediately
                // In a real app, you might want to make this asynchronous
                const colorData = ctx.getImageData(x, y, 1, 1).data;
                
                // Check if it's more blue (class 0) or more red (class 1)
                // This is a simplified approach - the actual color depends on your colormap
                return colorData[0] > colorData[2] ? 1 : 0; // Red > Blue = class 1, else class 0
            }
        }
        
        const info = results.model_info;
        
        // For linear SVM, use the weight vector and intercept for accurate prediction
        if (info.kernel === 'linear' && info.weights) {
            const weights = info.weights[0];
            const intercept = info.intercept[0];
            
            // w·x + b > 0 => class 1, else class 0
            return weights[0] * point.x + weights[1] * point.y + intercept > 0 ? 1 : 0;
        }
        
        // For non-linear kernels, implement best-effort prediction based on distance to support vectors
        // Use nearest neighbor approach with support vectors
        if (supportVectors && supportVectors.length > 0 && results.supportVectorClasses) {
            // Find the closest support vector
            let minDist = Infinity;
            let nearestSVIndex = -1;
            
            for (let i = 0; i < supportVectors.length; i++) {
                const sv = supportVectors[i];
                const dist = Math.sqrt(Math.pow(point.x - sv[0], 2) + Math.pow(point.y - sv[1], 2));
                if (dist < minDist) {
                    minDist = dist;
                    nearestSVIndex = i;
                }
            }
            
            // Use the class of the nearest support vector
            if (nearestSVIndex >= 0 && results.supportVectorClasses) {
                return results.supportVectorClasses[nearestSVIndex];
            }
        }
        
        // Implement kernel-specific logic
        if (kernel === 'rbf') {
            // Implement RBF kernel prediction
            let sum = 0;
            const gamma = parseFloat(hyperparams.gamma);
            
            // Use support vectors if available
            if (supportVectors && supportVectors.length > 0) {
                for (let i = 0; i < supportVectors.length; i++) {
                    const sv = supportVectors[i];
                    // RBF kernel function: exp(-gamma * ||x - sv||²)
                    const distance = Math.pow(point.x - sv[0], 2) + Math.pow(point.y - sv[1], 2);
                    sum += Math.exp(-gamma * distance);
                }
                // Threshold the sum for classification
                return sum > supportVectors.length / 2 ? 1 : 0;
            }
        }
        
        if (kernel === 'poly') {
            // Implement polynomial kernel 
            let sum = 0;
            const degree = parseInt(hyperparams.degree);
            const coef0 = parseFloat(hyperparams.coef0);
            
            if (supportVectors && supportVectors.length > 0) {
                for (let i = 0; i < supportVectors.length; i++) {
                    const sv = supportVectors[i];
                    // Poly kernel: (gamma * dot(x, sv) + coef0)^degree
                    const dot = point.x * sv[0] + point.y * sv[1];
                    sum += Math.pow(dot + coef0, degree);
                }
                return sum > 0 ? 1 : 0;
            }
        }
        
        // Last resort: use a simple estimate based on which side of the origin the point is on
        const isPositiveSide = (point.x > 0 && point.y > 0) || (point.x < 0 && point.y < 0);
        return isPositiveSide ? 1 : 0;
    };

    // Handle hyperparameter changes
    const handleHyperparamChange = (param, value) => {
        setHyperparams({
            ...hyperparams,
            [param]: value
        });
    };
   
    // Add the sample data modal JSX before the return statement
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
                width: '90%',
                maxWidth: '550px'
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
                            onClick={() => setSampleDataType('blobs')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'blobs' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'blobs' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>Blobs</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Distinct clusters</span>
                        </button>
                        <button
                            onClick={() => setSampleDataType('moons')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'moons' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'moons' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>Moons</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Curved boundaries</span>
                        </button>
                        <button
                            onClick={() => setSampleDataType('circles')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'circles' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'circles' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>Circles</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Concentric circles</span>
                        </button>
                    </div>
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
                        min="20"
                        max="200"
                        step="10"
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
                        <span>20 (Fewer points)</span>
                        <span>200 (More points)</span>
                    </div>
                </div>
                
                {/* Variance slider */}
                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="variance" style={{ 
                        display: 'block', 
                        marginBottom: '0.5rem', 
                        fontWeight: '500', 
                        color: '#4b5563' 
                    }}>
                        Variance: {sampleVariance.toFixed(1)}
                    </label>
                    <input
                        id="variance"
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={sampleVariance}
                        onChange={(e) => setSampleVariance(Number(e.target.value))}
                        style={{ width: '100%' }}
                    />
                    <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        fontSize: '0.8rem', 
                        color: '#6b7280',
                        marginTop: '0.25rem'
                    }}>
                        <span>0.1 (Compact)</span>
                        <span>1.0 (Spread out)</span>
                    </div>
                </div>
                
                {/* Clusters slider (only for blobs) */}
                {sampleDataType === 'blobs' && (
                    <div style={{ marginBottom: '1.25rem' }}>
                        <label htmlFor="clusters" style={{ 
                            display: 'block', 
                            marginBottom: '0.5rem', 
                            fontWeight: '500', 
                            color: '#4b5563' 
                        }}>
                            Number of Clusters: {sampleClusters}
                        </label>
                        <input
                            id="clusters"
                            type="range"
                            min="2"
                            max="4"
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
                            <span>4 clusters</span>
                        </div>
                    </div>
                )}
                
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
            {showSampleDataModal && <SampleDataModal />}
            
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
                        gridRow: '1 / 2',
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <div className="section-header">
                            <h2 className="section-title">Interactive SVM</h2>
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

                        <div style={{ marginBottom: '1rem' }}>
                            <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                                Click on the graph below to add data points. Current mode: <strong>{interactionMode === 'train' ? 'Training' : 'Testing'}</strong>
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
                            
                            {/* Hidden canvas for prediction */}
                            <canvas 
                                ref={tempCanvasRef}
                                width={canvasDimensions.width}
                                height={canvasDimensions.height}
                                style={{display: 'none'}}
                            />
                            
                            {/* Click overlay hint */}
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
                                Click to add {interactionMode === 'train' ? 'training' : 'testing'} point
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
                                    <strong>Training Points:</strong> {getValidPoints().length}
                                </div>
                                <div>
                                    <strong>Points to Predict:</strong> {predictedPoints.length}
                                </div>
                                <div>
                                    <strong>Support Vectors:</strong> {supportVectors.length}
                                </div>
                            </div>
                        </div>
                        
                        {/* Legend & How SVM Works */}
                        <div style={{ 
                            width: '100%',
                            backgroundColor: 'white', 
                            padding: '1rem', 
                            borderRadius: '6px', 
                            border: '1px solid #e5e7eb',
                            fontSize: '0.85rem'
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
                                    <span>Class 0</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ 
                                        width: '12px', 
                                        height: '12px', 
                                        borderRadius: '50%', 
                                        backgroundColor: 'rgba(239, 68, 68, 0.7)', 
                                        border: '1px solid #333' 
                                    }}></div>
                                    <span>Class 1</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ 
                                        width: '12px', 
                                        height: '12px', 
                                        backgroundColor: 'rgba(156, 163, 175, 0.7)',
                                        border: '1px solid #333',
                                        borderStyle: 'dashed'
                                    }}></div>
                                    <span>Unpredicted</span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ 
                                        width: '12px', 
                                        height: '12px', 
                                        borderRadius: '50%',
                                        backgroundColor: 'rgba(30, 80, 200, 0.85)',
                                        border: '3px solid #000000'
                                    }}></div>
                                    <span>Support Vector</span>
                                </div>
                            </div>
                            
                            <h3 style={{ marginTop: '1rem', marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>How SVM Works</h3>
                            <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                                <p style={{ marginBottom: '0.5rem' }}>
                                    Support Vector Machine works by finding the optimal hyperplane:
                                </p>
                                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                                    <li>Finds maximum margin between classes</li>
                                    <li>Support vectors are critical points</li>
                                    <li>Kernels transform data for nonlinear boundaries</li>
                                </ol>
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

                            {/* Margin Width parameter */}
                            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                                <label htmlFor="margin-width" style={{ 
                                    display: 'block', 
                                    marginBottom: '0.5rem', 
                                    fontWeight: '500', 
                                    color: '#4b5563',
                                    fontSize: '1rem'
                                }}>
                                    Margin Width: {hyperparams.marginWidth}
                                </label>
                                <input 
                                    id="margin-width"
                                    type="range" 
                                    min="0.1" 
                                    max="5.0" 
                                    step="0.1"
                                    value={hyperparams.marginWidth}
                                    onChange={(e) => handleHyperparamChange('marginWidth', e.target.value)}
                                    style={{ width: '100%' }}
                                />
                                <div style={{ 
                                    display: 'flex', 
                                    justifyContent: 'space-between', 
                                    fontSize: '0.85rem', 
                                    color: '#6b7280',
                                    marginTop: '0.5rem'
                                }}>
                                    <span>0.1 (Narrow)</span>
                                    <span>5.0 (Wide)</span>
                                </div>
                            </div>

                            {/* Gamma (for rbf, poly, sigmoid) */}
                            {(kernel === 'rbf' || kernel === 'poly' || kernel === 'sigmoid') && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="gamma-param" style={{ 
                                        display: 'block', 
                                        marginBottom: '0.5rem', 
                                        fontWeight: '500', 
                                        color: '#4b5563',
                                        fontSize: '1rem'
                                    }}>
                                        Gamma: {hyperparams.gamma}
                                    </label>
                                    <input 
                                        id="gamma-param"
                                        type="range" 
                                        min="0.01" 
                                        max="2.0" 
                                        step="0.01"
                                        value={hyperparams.gamma}
                                        onChange={(e) => handleHyperparamChange('gamma', e.target.value)}
                                        style={{ width: '100%' }}
                                    />
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        fontSize: '0.85rem', 
                                        color: '#6b7280',
                                        marginTop: '0.5rem'
                                    }}>
                                        <span>0.01 (Lower influence)</span>
                                        <span>2.0 (Higher influence)</span>
                                    </div>
                                </div>
                            )}

                            {/* Degree (for poly) */}
                            {kernel === 'poly' && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="degree-param" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>
                                        Polynomial Degree: {hyperparams.degree}
                                    </label>
                                    <input 
                                        id="degree-param"
                                        type="range" 
                                        min="1" 
                                        max="10" 
                                        step="1"
                                        value={hyperparams.degree}
                                        onChange={(e) => handleHyperparamChange('degree', e.target.value)}
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
                                        <span>10 (Complex)</span>
                                    </div>
                                </div>
                            )}

                            {/* Coef0 (for poly and sigmoid) */}
                            {(kernel === 'poly' || kernel === 'sigmoid') && (
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label htmlFor="coef0-param" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>
                                        Coef0: {hyperparams.coef0}
                                    </label>
                                    <input 
                                        id="coef0-param"
                                        type="range" 
                                        min="0" 
                                        max="5" 
                                        step="0.1"
                                        value={hyperparams.coef0}
                                        onChange={(e) => handleHyperparamChange('coef0', e.target.value)}
                                        style={{ width: '100%' }}
                                    />
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        fontSize: '0.85rem', 
                                        color: '#6b7280',
                                        marginTop: '0.5rem'
                                    }}>
                                        <span>0 (Zero intercept)</span>
                                        <span>5 (High intercept)</span>
                                    </div>
                                </div>
                            )}
                            
                            {/* Show Support Vectors Toggle */}
                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={{ 
                                    display: 'flex', 
                                    alignItems: 'center',
                                    cursor: 'pointer',
                                    fontWeight: '500', 
                                    color: '#4b5563'
                                }}>
                                    <input 
                                        type="checkbox"
                                        checked={showSupportVectors}
                                        onChange={() => setShowSupportVectors(!showSupportVectors)}
                                        style={{ marginRight: '0.5rem' }}
                                    />
                                    Show Support Vectors
                                </label>
                            </div>
                            
                            <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>
                                Interaction Mode
                            </h3>
                            
                            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
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
                                    Training Points
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
                                    Prediction Points
                                </button>
                            </div>
                            
                            {interactionMode === 'train' && (
                                <div style={{ marginBottom: '0.75rem' }}>
                                    <p style={{ marginBottom: '0.75rem', color: '#4b5563', fontSize: '1rem' }}>
                                        Select class for training points:
                                    </p>
                                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                                        <button
                                            onClick={() => setCurrentClass(0)}
                                            style={{
                                                padding: '0.75rem 0.5rem',
                                                border: 'none',
                                                borderRadius: '0.5rem',
                                                backgroundColor: currentClass === 0 ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)',
                                                color: currentClass === 0 ? 'white' : '#1e40af',
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
                                                padding: '0.75rem 0.5rem',
                                                border: 'none',
                                                borderRadius: '0.5rem',
                                                backgroundColor: currentClass === 1 ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)',
                                                color: currentClass === 1 ? 'white' : '#b91c1c',
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
                                    marginBottom: '1.25rem'
                                }}
                            >
                                {loading ? 'Running...' : 'Run SVM'}
                            </button>

                            {/* Predict Button */}
                            <button 
                                onClick={handlePredict}
                                disabled={loading || backendStatus === "disconnected" || !results}
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
                                    opacity: (loading || backendStatus === "disconnected" || !results) ? 0.7 : 1
                                }}
                            >
                                {loading ? 'Predicting...' : 'Predict Points'}
                            </button>
                        </div>

                        {/* Results Section */}
                        {results && (
                            <div style={{ 
                                width: '100%',
                                backgroundColor: 'white', 
                                padding: '1.5rem', 
                                borderRadius: '6px', 
                                border: '1px solid #e5e7eb',
                                fontSize: '0.95rem'
                            }}>
                                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Results</h3>
                                
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
                                        Accuracy: {(results.accuracy * 100).toFixed(2)}%
                                    </p>
                                    <p>Support Vectors: {supportVectors.length}</p>
                                    <p>Kernel: {kernel}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
            
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

export default SVM;
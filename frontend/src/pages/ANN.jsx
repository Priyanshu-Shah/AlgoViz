import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { checkHealth } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function ANN() {
    const navigate = useNavigate();
    const [dataPairs, setDataPairs] = useState([{ x: '', y: '', class: 0 }]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState("checking");
    const [sampleLoading, setSampleLoading] = useState(false);
    const [currentClass, setCurrentClass] = useState(0);
    const [interactionMode, setInteractionMode] = useState('train');
    const [predictedPoints, setPredictedPoints] = useState([]);
    const [showNeuralNetwork, setShowNeuralNetwork] = useState(true);
    const [showSampleDataModal, setShowSampleDataModal] = useState(false);
    const [sampleDataType, setSampleDataType] = useState('classification');
    const [sampleCount, setSampleCount] = useState(40);
    const [sampleClusters, setSampleClusters] = useState(2);
    const [sampleVariance, setSampleVariance] = useState(0.5);
    const [decisionBoundary, setDecisionBoundary] = useState(null);
    const [neuronVisualizations, setNeuronVisualizations] = useState({});
    
    
    // Neural network architecture
    const [networkArchitecture, setNetworkArchitecture] = useState({
        hiddenLayers: [
            { neurons: 8, activation: 'relu' },
            { neurons: 8, activation: 'relu' }
        ],
        learningRate: 0.005,
        epochs: 200,
        batchSize: 16,
        activation: 'relu',
        outputActivation: 'softmax'
    });

    // Canvas ref and state for interactive plotting
    const canvasRef = useRef(null);
    const networkCanvasRef = useRef(null);
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
    
    // Handle Neural Network layer changes
    const handleAddLayer = () => {
        const updatedLayers = [...networkArchitecture.hiddenLayers, { neurons: 5, activation: 'relu' }];
        setNetworkArchitecture({ ...networkArchitecture, hiddenLayers: updatedLayers });
    };


    const handleAddLayerAt = (index) => {
        const updatedLayers = [...networkArchitecture.hiddenLayers];
        // Insert new layer at the specified index
        updatedLayers.splice(index + 1, 0, { neurons: 5, activation: 'relu' });
        setNetworkArchitecture({ ...networkArchitecture, hiddenLayers: updatedLayers });
    };
    
    const handleRemoveLayerAt = (index) => {
        if (networkArchitecture.hiddenLayers.length <= 1) return;
        
        const updatedLayers = [...networkArchitecture.hiddenLayers];
        updatedLayers.splice(index, 1);
        setNetworkArchitecture({ ...networkArchitecture, hiddenLayers: updatedLayers });
    };

    const handleRemoveLayer = () => {
        if (networkArchitecture.hiddenLayers.length > 1) {
            const updatedLayers = [...networkArchitecture.hiddenLayers];
            updatedLayers.pop();
            setNetworkArchitecture({ ...networkArchitecture, hiddenLayers: updatedLayers });
        }
    };

    const handleLayerChange = (index, field, value) => {
        const updatedLayers = [...networkArchitecture.hiddenLayers];
        
        if (field === 'neurons') {
            // Ensure neurons value is a positive integer between 1 and 20
            const neuronsValue = Math.max(1, Math.min(20, parseInt(value) || 1));
            updatedLayers[index][field] = neuronsValue;
        } else {
            updatedLayers[index][field] = value;
        }
        
        setNetworkArchitecture({ ...networkArchitecture, hiddenLayers: updatedLayers });
    };

    const handleHyperparamChange = (param, value) => {
        if (param === 'learningRate') {
            // Ensure learning rate is positive and within a valid range
            value = Math.max(0.0001, Math.min(1, parseFloat(value)));
        } else if (param === 'epochs' || param === 'batchSize') {
            // Ensure epochs and batch size are positive integers
            value = Math.max(1, parseInt(value) || 1);
        }
        
        setNetworkArchitecture({
            ...networkArchitecture,
            [param]: value
        });
    };

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

    const drawPoints = (ctx, canvas, points) => {
        if (!points || points.length === 0) return;
        
        points.forEach(point => {
            // Convert data coordinates to screen coordinates
            const x = ((parseFloat(point.x) - scale.x.min) / (scale.x.max - scale.x.min)) * canvas.width;
            const y = canvas.height - ((parseFloat(point.y) - scale.y.min) / (scale.y.max - scale.y.min)) * canvas.height;
            
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            
            // Determine fill color based on point type and class
            if (point.isPredicted) {
                if (point.predictedClass !== undefined) {
                    // After prediction, color based on predicted class
                    if (point.predictedClass === 0) {
                        ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue for class 0
                        ctx.strokeStyle = '#1e40af';
                    } else if (point.predictedClass === 1) {
                        ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red for class 1
                        ctx.strokeStyle = '#b91c1c';
                    } else if (point.predictedClass === 2) {
                        ctx.fillStyle = 'rgba(34, 197, 94, 0.7)'; // Green for class 2
                        ctx.strokeStyle = '#15803d';
                    } else {
                        ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray for unknown class
                        ctx.strokeStyle = '#4b5563';
                    }
                } else {
                    // Before prediction, use grey for unpredicted points
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
                    ctx.strokeStyle = '#4b5563';
                }
                // Use dashed outline for prediction points
                ctx.setLineDash([2, 2]);
            } else {
                // Training points
                if (point.class === 0) {
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue for class 0
                    ctx.strokeStyle = '#1e40af';
                } else if (point.class === 1) {
                    ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red for class 1
                    ctx.strokeStyle = '#b91c1c';
                } else if (point.class === 2) {
                    ctx.fillStyle = 'rgba(34, 197, 94, 0.7)'; // Green for class 2
                    ctx.strokeStyle = '#15803d';
                } else {
                    ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray fallback
                    ctx.strokeStyle = '#4b5563';
                }
                ctx.setLineDash([]);
            }
            
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]);
        });
    };

    // Draw neural network visualization
    const drawNeuralNetwork = () => {
        const canvas = networkCanvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
        
        // Create full network structure including input and output layers
        const fullNetwork = [
            { neurons: 2, activation: 'input' },  // Input layer: x, y coordinates
            ...networkArchitecture.hiddenLayers,
            { neurons: 3, activation: networkArchitecture.outputActivation }  // Output layer with 3 neurons
        ];
        
        const numLayers = fullNetwork.length;
        const layerGap = width / (numLayers + 1);
        const networkHeight = height * 0.8;
        const verticalOffset = height * 0.1;
        
        // Draw connections first (so they appear behind neurons)
        ctx.strokeStyle = '#d1d5db';
        ctx.lineWidth = 1;
        
        for (let i = 0; i < numLayers - 1; i++) {
            const currentLayer = fullNetwork[i];
            const nextLayer = fullNetwork[i + 1];
            
            const currentX = layerGap * (i + 1);
            const nextX = layerGap * (i + 2);
            
            const currentNodeGap = networkHeight / (Math.max(currentLayer.neurons, 2) + 1);
            const nextNodeGap = networkHeight / (Math.max(nextLayer.neurons, 2) + 1);
            
            for (let j = 0; j < currentLayer.neurons; j++) {
                const currentY = verticalOffset + currentNodeGap * (j + 1);
                
                for (let k = 0; k < nextLayer.neurons; k++) {
                    const nextY = verticalOffset + nextNodeGap * (k + 1);
                    
                    ctx.beginPath();
                    ctx.moveTo(currentX, currentY);
                    ctx.lineTo(nextX, nextY);
                    ctx.stroke();
                }
            }
        }
        
        // Draw neurons
        for (let i = 0; i < numLayers; i++) {
            const layer = fullNetwork[i];
            const layerX = layerGap * (i + 1);
            const nodeGap = networkHeight / (Math.max(layer.neurons, 2) + 1);
            
            // Determine color based on activation function
            let fillColor;
            switch (layer.activation) {
                case 'relu':
                    fillColor = 'rgba(59, 130, 246, 0.7)';  // Blue
                    break;
                case 'tanh':
                    fillColor = 'rgba(139, 92, 246, 0.7)';  // Purple
                    break;
                case 'sigmoid':
                    fillColor = 'rgba(34, 197, 94, 0.7)';   // Green
                    break;
                case 'input':
                    fillColor = 'rgba(249, 115, 22, 0.7)';  // Orange
                    break;
                default:
                    fillColor = 'rgba(107, 114, 128, 0.7)'; // Gray
            }
            
            // Draw each neuron
            for (let j = 0; j < layer.neurons; j++) {
                const nodeY = verticalOffset + nodeGap * (j + 1);
                
                ctx.beginPath();
                ctx.arc(layerX, nodeY, 12, 0, Math.PI * 2);
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            
            // Add layer labels
            ctx.fillStyle = '#4b5563';
            ctx.font = '12px Inter, sans-serif';
            
            const labelText = i === 0 ? 'Input' : 
                              i === numLayers - 1 ? 'Output' :
                              `Hidden ${i}`;
            
            ctx.textAlign = 'center';
            ctx.fillText(labelText, layerX, height - 10);
            
            // Add activation name for hidden and output layers
            if (i > 0) {
                ctx.fillText(layer.activation, layerX, height - 25);
            }
        }
    };

    // Draw neural network on initialization and when architecture changes
    useEffect(() => {
        if (showNeuralNetwork) {
            drawNeuralNetwork();
        }
    }, [networkArchitecture, showNeuralNetwork]);
        
    // Convert screen coordinates to data coordinates
    const screenToData = (x, y) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        
        const dataX = scale.x.min + (x / canvas.width) * (scale.x.max - scale.x.min);
        const dataY = scale.y.min + ((canvas.height - y) / canvas.height) * (scale.y.max - scale.y.min);
        
        return { x: dataX, y: dataY };
    };

    // Modify the useEffect for canvas drawing
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
        
        // Just draw points - don't show decision boundary on main canvas
        drawPoints(ctx, canvas, [...getValidPoints(), ...predictedPoints]);
    }, [dataPairs, predictedPoints]);

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

    // Update the loadSampleData function to show the modal
    const loadSampleData = () => {
        setShowSampleDataModal(true);
    };

    // Add a function to generate sample data with options
    const generateSampleDataWithOptions = () => {
        setLoading(true);
        setShowSampleDataModal(false);
        
        // Map frontend dataset type names to backend names
        const backendDatasetType = {
            'xor': 'xor',
            'circle': 'circle',
            'spiral': 'spiral'
        }[sampleDataType] || 'blobs';
        
        // Call the backend API
        axios.post(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/api/ann/sample_data`, {
            dataset_type: backendDatasetType,
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
            }
        })
        .catch(err => {
            console.error("Error generating sample data:", err);
            setError("Failed to generate sample data: " + (err.response?.data?.error || err.message));
            
            // Only as a last resort, create minimal default data
            const defaultData = generateDefaultSampleData();
            setDataPairs(defaultData);
        })
        .finally(() => {
            setLoading(false);
        });
    };
    
    const generateDefaultSampleData = () => {
        // Instead of hard-coding data, show an error message
        setError("Failed to fetch sample data. Please try again.");
        
        // Return a minimal set of points that at least won't crash the app
        return [
            { x: '-2', y: '-2', class: 0 },
            { x: '2', y: '2', class: 1 }
        ];
    };

    const resetData = () => {
        setDataPairs([{ x: '', y: '', class: 0 }]);
        setResults(null);
        setError(null);
        setPredictedPoints([]);
        setDecisionBoundary(null);
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
            setError('Please add at least 2 data points for training the neural network.');
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
        setPredictedPoints([]);
        setResults(null);
    
        try {
            // Prepare network architecture for the API
            const networkConfig = {
                hidden_layers: networkArchitecture.hiddenLayers.map(layer => ({
                    neurons: layer.neurons,
                    activation: layer.activation
                })),
                learning_rate: networkArchitecture.learningRate,
                epochs: parseInt(networkArchitecture.epochs),
                batch_size: networkArchitecture.batchSize,
                activation: networkArchitecture.activation,
                output_activation: networkArchitecture.outputActivation
            };
            
            // Prepare data for the API
            const apiData = {
                X: validPairs.map(pair => [parseFloat(pair.x), parseFloat(pair.y)]),
                y: validPairs.map(pair => parseInt(pair.class)),
                network_config: networkConfig
            };
    
            console.log('Calling API endpoint with data:', apiData);
            const response = await axios.post(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/api/ann/train`, apiData);
    
            if (response.data.error) {
                throw new Error(response.data.error);
            }
    
            console.log('Results state set to:', response.data);
            
            // Handle response
            setResults(response.data);
            
            // Handle the decision boundary image
            if (response.data.decision_boundary) {
                setDecisionBoundary(response.data.decision_boundary);
            }
            if (response.data.neuron_visualizations) {
                setNeuronVisualizations(response.data.neuron_visualizations);
            }
            
        } catch (err) {
            console.error('Error details:', err);
            setError(`Error: ${err.message || 'An error occurred while running the model.'}`);
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = async () => {
        if (!results) {
            setError('Please train the neural network first before making predictions.');
            return;
        }
    
        if (predictedPoints.length === 0) {
            setError('Please add at least one point to predict by clicking on the canvas in testing mode.');
            return;
        }
    
        setError(null);
        setLoading(true);
    
        try {
            // Prepare data for the API
            const apiData = {
                model: results, 
                points: predictedPoints.map(point => [parseFloat(point.x), parseFloat(point.y)])
            };
    
            const response = await axios.post('http://localhost:5000/api/ann/predict', apiData);
    
            if (response.data.error) {
                throw new Error(response.data.error);
            }
    
            // Update predicted points with their predicted classes
            const newPredictedPoints = [...predictedPoints];
            response.data.predictions.forEach((pred, index) => {
                if (index < newPredictedPoints.length) {
                    // For multi-class classification, the prediction will be the class index
                    let predictedClass = parseInt(pred);
                    newPredictedPoints[index].predictedClass = predictedClass;
                    newPredictedPoints[index].rawPrediction = pred;
                }
            });
    
            setPredictedPoints(newPredictedPoints);
        } catch (err) {
            setError(`Error during prediction: ${err.message}`);
            console.error(err);
        } finally {
            setLoading(false);
        }
    };
   
    // Add the sample data modal JSX
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
                            onClick={() => setSampleDataType('xor')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'xor' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'xor' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>XOR</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Non-linear pattern</span>
                        </button>
                        <button
                            onClick={() => setSampleDataType('circle')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'circle' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'circle' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>Circle</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Radial pattern</span>
                        </button>
                        <button
                            onClick={() => setSampleDataType('spiral')}
                            style={{
                                padding: '0.5rem 0.75rem',
                                backgroundColor: sampleDataType === 'spiral' ? '#3b82f6' : '#e5e7eb',
                                color: sampleDataType === 'spiral' ? 'white' : '#4b5563',
                                border: 'none',
                                borderRadius: '0.5rem',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center'
                            }}
                        >
                            <span style={{ fontWeight: '500' }}>Spiral</span>
                            <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Complex pattern</span>
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
                
                {/* Noise slider */}
                <div style={{ marginBottom: '1.25rem' }}>
                    <label htmlFor="variance" style={{ 
                        display: 'block', 
                        marginBottom: '0.5rem', 
                        fontWeight: '500', 
                        color: '#4b5563' 
                    }}>
                        Noise Level: {sampleVariance.toFixed(1)}
                    </label>
                    <input
                        id="variance"
                        type="range"
                        min="0.0"
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
                        <span>0.0 (Clean)</span>
                        <span>1.0 (Noisy)</span>
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
                {showSampleDataModal && <SampleDataModal />}
                
                <div className="model-header">
                    <button className="back-button" onClick={() => navigate('/')}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                            <path d="M12 19L5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                        <span style={{ marginLeft: '0.5rem' }}>Back to Home</span>
                    </button>
                    <h1 className="model-title">Neural Network</h1>
                </div>

                <p className="model-description">
                    Artificial Neural Networks are computing systems inspired by biological neural networks. They can learn to perform tasks by considering examples, without being explicitly programmed.
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
                        {/* Left column: Interactive Plot and Neural Network Visualization */}
                        <div style={{ 
                            width: '100%',
                            gridColumn: '1 / 2',
                            gridRow: '1 / 2',
                            display: 'flex',
                            flexDirection: 'column'
                        }}>
                            <div className="section-header">
                                <h2 className="section-title">Interactive Neural Network</h2>
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
                                    Dataset Statistics:
                                </p>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                    <div>
                                        <strong>Training Points:</strong> {getValidPoints().length}
                                    </div>
                                    <div>
                                        <strong>Points to Predict:</strong> {predictedPoints.length}
                                    </div>
                                    <div>
                                        <strong>Classes:</strong> 3
                                    </div>
                                </div>
                            </div>
                            
                            {/* Neural Network Visualization */}
                            <div style={{
                                position: 'relative',
                                marginBottom: '1rem',
                                border: '1px solid #e5e7eb',
                                borderRadius: '0.75rem',
                                overflow: 'hidden',
                                backgroundColor: '#ffffff',
                                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
                                width: '100%',
                                height: 350
                            }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center'
                                }}>
                                    {showNeuralNetwork && (
                                        <canvas
                                            ref={networkCanvasRef}
                                            width={700}
                                            height={350}
                                            style={{
                                                width: '100%',
                                                height: '100%'
                                            }}
                                        />
                                    )}
                                    
                                    {!showNeuralNetwork && (
                                        <div style={{ textAlign: 'center' }}>
                                            <p>Neural network visualization is hidden</p>
                                            <button
                                                onClick={() => setShowNeuralNetwork(true)}
                                                style={{
                                                    padding: '0.5rem 1rem',
                                                    backgroundColor: '#3b82f6',
                                                    color: 'white',
                                                    border: 'none',
                                                    borderRadius: '0.5rem',
                                                    cursor: 'pointer',
                                                    marginTop: '1rem'
                                                }}
                                            >
                                                Show Network
                                            </button>
                                        </div>
                                    )}
                                </div>
                                
                                {showNeuralNetwork && (
                                    <div style={{
                                        position: 'absolute',
                                        top: '10px',
                                        right: '10px'
                                    }}>
                                        <button
                                            onClick={() => setShowNeuralNetwork(false)}
                                            style={{
                                                padding: '4px 8px',
                                                backgroundColor: '#f3f4f6',
                                                color: '#4b5563',
                                                border: 'none',
                                                borderRadius: '4px',
                                                fontSize: '0.8rem',
                                                cursor: 'pointer'
                                            }}
                                        >
                                            Hide
                                        </button>
                                    </div>
                                )}
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
                            
                            {/* Network Architecture Controls */}
                            <div style={{ 
                                marginBottom: '1.5rem', 
                                backgroundColor: 'white', 
                                padding: '1.5rem', 
                                borderRadius: '6px', 
                                border: '1px solid #e5e7eb',
                                width: '100%'
                            }}>
                                <h3 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: '500' }}>Network Architecture</h3>
                                
                                {/* Hidden Layer Controls */}
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        marginBottom: '0.75rem'
                                    }}>
                                        <label style={{ fontWeight: '500', color: '#4b5563' }}>
                                            Hidden Layers
                                        </label>
                                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                                            <button
                                                onClick={handleAddLayer}
                                                style={{
                                                    padding: '0.25rem 0.5rem',
                                                    backgroundColor: '#f3f4f6',
                                                    color: '#4b5563',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    fontSize: '0.9rem',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                + Add
                                            </button>
                                            <button
                                                onClick={handleRemoveLayer}
                                                disabled={networkArchitecture.hiddenLayers.length <= 1}
                                                style={{
                                                    padding: '0.25rem 0.5rem',
                                                    backgroundColor: networkArchitecture.hiddenLayers.length <= 1 ? '#f3f4f6' : '#fee2e2',
                                                    color: networkArchitecture.hiddenLayers.length <= 1 ? '#9ca3af' : '#b91c1c',
                                                    border: 'none',
                                                    borderRadius: '4px',
                                                    fontSize: '0.9rem',
                                                    cursor: networkArchitecture.hiddenLayers.length <= 1 ? 'not-allowed' : 'pointer'
                                                }}
                                            >
                                                - Remove
                                            </button>
                                        </div>
                                    </div>
                                    
                                    {/* Layer configuration UI */}
                                    <div style={{ 
                                        display: 'flex', 
                                        flexDirection: 'column',
                                        gap: '0.75rem',
                                        maxHeight: '200px',
                                        overflowY: 'auto',
                                        padding: '0.5rem',
                                        border: '1px solid #e5e7eb',
                                        borderRadius: '0.5rem',
                                        backgroundColor: '#f9fafb'
                                    }}>
                                        {networkArchitecture.hiddenLayers.map((layer, index) => (
                                            <div 
                                                key={index} 
                                                style={{ 
                                                    display: 'flex', 
                                                    gap: '0.75rem',
                                                    padding: '0.5rem',
                                                    backgroundColor: 'white',
                                                    border: '1px solid #e5e7eb',
                                                    borderRadius: '0.5rem'
                                                }}
                                            >
                                                <div style={{ flex: 1 }}>
                                                    <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '0.25rem', color: '#6b7280' }}>
                                                        Neurons
                                                    </label>
                                                    <input
                                                        type="number"
                                                        min="1"
                                                        max="20"
                                                        value={layer.neurons}
                                                        onChange={(e) => handleLayerChange(index, 'neurons', e.target.value)}
                                                        style={{
                                                            width: '100%',
                                                            padding: '0.25rem 0.5rem',
                                                            border: '1px solid #d1d5db',
                                                            borderRadius: '0.25rem'
                                                        }}
                                                    />
                                                </div>
                                                <div style={{ flex: 2 }}>
                                                    <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '0.25rem', color: '#6b7280' }}>
                                                        Activation
                                                    </label>
                                                    <select
                                                        value={layer.activation}
                                                        onChange={(e) => handleLayerChange(index, 'activation', e.target.value)}
                                                        style={{
                                                            width: '100%',
                                                            padding: '0.25rem 0.5rem',
                                                            border: '1px solid #d1d5db',
                                                            borderRadius: '0.25rem'
                                                        }}
                                                    >
                                                        <option value="relu">ReLU</option>
                                                        <option value="sigmoid">Sigmoid</option>
                                                        <option value="tanh">Tanh</option>
                                                    </select>
                                                </div>
                                                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: '0.25rem' }}>
                                                    <button
                                                        onClick={() => handleAddLayerAt(index)}
                                                        title="Add layer after this one"
                                                        style={{
                                                            padding: '0.25rem',
                                                            backgroundColor: '#e5e7eb',
                                                            color: '#4b5563',
                                                            border: 'none',
                                                            borderRadius: '4px',
                                                            cursor: 'pointer',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center'
                                                        }}
                                                    >
                                                        <span style={{ fontSize: '1rem' }}>+</span>
                                                    </button>
                                                    <button
                                                        onClick={() => handleRemoveLayerAt(index)}
                                                        disabled={networkArchitecture.hiddenLayers.length <= 1}
                                                        title="Remove this layer"
                                                        style={{
                                                            padding: '0.25rem',
                                                            backgroundColor: networkArchitecture.hiddenLayers.length <= 1 ? '#f3f4f6' : '#fee2e2',
                                                            color: networkArchitecture.hiddenLayers.length <= 1 ? '#9ca3af' : '#b91c1c',
                                                            border: 'none',
                                                            borderRadius: '4px',
                                                            cursor: networkArchitecture.hiddenLayers.length <= 1 ? 'not-allowed' : 'pointer',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center'
                                                        }}
                                                    >
                                                        <span style={{ fontSize: '1rem' }}>-</span>
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                
                                {/* Training Parameters */}
                                <h4 style={{ marginBottom: '0.75rem', fontSize: '1rem', fontWeight: '500', color: '#4b5563' }}>
                                    Training Parameters
                                </h4>
                                
                                {/* Learning Rate */}
                                <div style={{ marginBottom: '1rem' }}>
                                    <label style={{ 
                                        display: 'block', 
                                        marginBottom: '0.5rem', 
                                        fontSize: '0.9rem',
                                        color: '#4b5563' 
                                    }}>
                                        Learning Rate: {networkArchitecture.learningRate}
                                    </label>
                                    <input 
                                        type="range" 
                                        min="0.0001" 
                                        max="0.1" 
                                        step="0.001"
                                        value={networkArchitecture.learningRate}
                                        onChange={(e) => handleHyperparamChange('learningRate', e.target.value)}
                                        style={{ width: '100%' }}
                                    />
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        fontSize: '0.8rem', 
                                        color: '#6b7280'
                                    }}>
                                        <span>0.0001 (Slow)</span>
                                        <span>0.1 (Fast)</span>
                                    </div>
                                </div>

                                {/* Batch Size */}
                                <div style={{ marginBottom: '1rem' }}>
                                    <label style={{ 
                                        display: 'block', 
                                        marginBottom: '0.5rem',
                                        fontSize: '0.9rem',
                                        color: '#4b5563' 
                                    }}>
                                        Batch Size: {networkArchitecture.batchSize}
                                    </label>
                                    <input 
                                        type="range" 
                                        min="1" 
                                        max="64" 
                                        step="1"
                                        value={networkArchitecture.batchSize}
                                        onChange={(e) => handleHyperparamChange('batchSize', e.target.value)}
                                        style={{ width: '100%' }}
                                    />
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        fontSize: '0.8rem', 
                                        color: '#6b7280'
                                    }}>
                                        <span>1 (Slow, accurate)</span>
                                        <span>64 (Fast, generalized)</span>
                                    </div>
                                </div>
                                
                                {/* Epochs */}
                                <div style={{ marginBottom: '1rem' }}>
                                    <label style={{ 
                                        display: 'block', 
                                        marginBottom: '0.5rem',
                                        fontSize: '0.9rem',
                                        color: '#4b5563' 
                                    }}>
                                        Epochs: {networkArchitecture.epochs}
                                    </label>
                                    <input 
                                        type="range" 
                                        min="10" 
                                        max="1000" 
                                        step="10"
                                        value={networkArchitecture.epochs}
                                        onChange={(e) => handleHyperparamChange('epochs', e.target.value)}
                                        style={{ width: '100%' }}
                                    />
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        fontSize: '0.8rem', 
                                        color: '#6b7280'
                                    }}>
                                        <span>10</span>
                                        <span>1000</span>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Interaction Mode Controls */}
                            <div style={{ 
                                marginBottom: '1.5rem', 
                                backgroundColor: 'white', 
                                padding: '1.5rem', 
                                borderRadius: '6px', 
                                border: '1px solid #e5e7eb',
                                width: '100%'
                            }}>
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
                                        <div style={{ display: 'flex', gap: '0.5rem' }}>
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
                                            <button
                                                onClick={() => setCurrentClass(2)}
                                                style={{
                                                    padding: '0.75rem 0.5rem',
                                                    border: 'none',
                                                    borderRadius: '0.5rem',
                                                    backgroundColor: currentClass === 2 ? 'rgba(34, 197, 94, 1)' : 'rgba(34, 197, 94, 0.1)',
                                                    color: currentClass === 2 ? 'white' : '#15803d',
                                                    cursor: 'pointer',
                                                    fontWeight: '500',
                                                    flex: 1,
                                                    fontSize: '0.95rem'
                                                }}
                                            >
                                                Class 2
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
                                
                                {/* Train Button */}
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
                                    {loading ? 'Training...' : 'Train Network'}
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
                                        <p>Loss: {results.loss?.toFixed(4) || 'N/A'}</p>
                                        <p>Epochs: {results.epochs}</p>
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

                                    {results && results.model_info && (
                                        <div style={{ marginTop: '1rem' }}>
                                            <h4 style={{ marginBottom: '0.5rem' }}>Training History:</h4>
                                            <div style={{ height: 200, width: '100%' }}>
                                                <ResponsiveContainer>
                                                    <LineChart
                                                        data={results.model_info.training_history || []}
                                                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                                    >
                                                        <CartesianGrid strokeDasharray="3 3" />
                                                        <XAxis dataKey="epoch" />
                                                        <YAxis />
                                                        <Tooltip />
                                                        <Legend />
                                                        <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Training Accuracy" />
                                                        <Line type="monotone" dataKey="val_accuracy" stroke="#82ca9d" name="Validation Accuracy" />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    )}
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

export default ANN;
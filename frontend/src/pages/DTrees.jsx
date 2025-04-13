import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import './ModelPage.css';

function DTrees() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('classification');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connected");
  
  // States for visualization
  const canvasRef = useRef(null);
  const [selectedClass, setSelectedClass] = useState('1');
  const [pointsMode, setPointsMode] = useState('train'); // 'train' or 'predict'
  const [trainingPoints, setTrainingPoints] = useState([]);
  const [predictPoints, setPredictPoints] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [isTreeEnlarged, setIsTreeEnlarged] = useState(false);
  // For sample data generation modal
const [showSampleDataModal, setShowSampleDataModal] = useState(false);
const [sampleDataType, setSampleDataType] = useState('blobs');
const [sampleCount, setSampleCount] = useState(40);
const [sampleClusters, setSampleClusters] = useState(3);
const [sampleVariance, setSampleVariance] = useState(0.5);
const [sampleSparsity, setSampleSparsity] = useState(1.0);

// Add this handler function
const toggleTreeEnlargement = () => {
  setIsTreeEnlarged(!isTreeEnlarged);
};
  // Use useMemo to wrap objects used in dependencies
  const canvasDimensions = useMemo(() => ({ width: 600, height: 600 }), []); // Square canvas for proper aspect ratio
  
  // Scale: Use -8 to 8 range to match KNN
  const scale = useMemo(() => ({
    x: { min: -8, max: 8 },
    y: { min: -8, max: 8 }
  }), []);
  
  // For point input dialog
  const [showPointDialog, setShowPointDialog] = useState(false);
  const [dialogPosition, setDialogPosition] = useState({ x: 0, y: 0 });
  const [tempPoint, setTempPoint] = useState(null);
  const [tempValue, setTempValue] = useState('');
  
  // For decision tree parameters
  const [maxDepth, setMaxDepth] = useState(3);
  const [minSamplesSplit, setMinSamplesSplit] = useState(2);
  const [criterion, setCriterion] = useState('gini');
  
  // For visualization
  const [showTree, setShowTree] = useState(false);
  const [treeImg, setTreeImg] = useState(null);
  const [decisionBoundaryImg, setDecisionBoundaryImg] = useState(null);
  
  // For prediction animation
  const [animatingPrediction, setAnimatingPrediction] = useState(false);
  const [currentAnimationPath, setCurrentAnimationPath] = useState([]);
  const [currentAnimationStep, setCurrentAnimationStep] = useState(0);
  const [currentHighlightedNode, setCurrentHighlightedNode] = useState(null);
  
  // For hover effect on visualization
  const [hoveredPoint, setHoveredPoint] = useState(null);
  
  const apiUrl = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000';

  // Check backend health when component mounts
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${apiUrl}/health`);
        setBackendStatus(response.data.status === "healthy" ? "connected" : "disconnected");
      } catch (err) {
        console.error("Backend health check failed:", err);
        setBackendStatus("disconnected");
      }
    };
    
    checkBackendHealth();
  }, [apiUrl]);
  
  // Handle tab change
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    resetData();
  };

  // Now modify the generateSampleData to show the modal
  const generateSampleData = () => {
    setShowSampleDataModal(true);
  };
 
    // Add this new function to handle sample data generation with options
    const generateSampleDataWithOptions = () => {
      setLoading(true);
      setShowSampleDataModal(false);
      
      axios.post(`${apiUrl}/dtree/sample_data`, { 
        type: activeTab,
        dataset_type: sampleDataType,
        count: sampleCount,
        n_clusters: sampleClusters,
        variance: sampleVariance,
        sparsity: sampleSparsity
      })
      .then(response => {
        if (response.data && response.data.points) {
          setTrainingPoints(response.data.points);
          
          // Reset other states
          setPredictPoints([]);
          setPredictions([]);
          setPointsMode('train');
          setShowTree(false);
          setTreeImg(null);
          setDecisionBoundaryImg(null);
        }
      })
      .catch(err => {
        console.error("Error generating sample data:", err);
        setError("Failed to generate sample data: " + (err.response?.data?.error || err.message));
        
        // Fallback to client-side sample data generation if server fails
        generateSampleDataLocally();
      })
      .finally(() => {
        setLoading(false);
      });
    };
  
  // Fallback local data generation
  const generateSampleDataLocally = () => {
    if (activeTab === 'classification') {
      // Generate classification data
      const newPoints = [];
      // Class 1: cluster around (-4, -4)
      for (let i = 0; i < 12; i++) {
        newPoints.push({
          x1: -4 + (Math.random() * 3 - 1.5),
          x2: -4 + (Math.random() * 3 - 1.5),
          y: '0'
        });
      }
      // Class 2: cluster around (4, 4)
      for (let i = 0; i < 12; i++) {
        newPoints.push({
          x1: 4 + (Math.random() * 3 - 1.5),
          x2: 4 + (Math.random() * 3 - 1.5),
          y: '1'
        });
      }
      // Class 3: cluster around (4, -4)
      for (let i = 0; i < 12; i++) {
        newPoints.push({
          x1: 4 + (Math.random() * 3 - 1.5),
          x2: -4 + (Math.random() * 3 - 1.5),
          y: '2'
        });
      }
      setTrainingPoints(newPoints);
    } else {
      // Generate regression data (linear trend with noise)
      const newPoints = [];
      for (let i = 0; i < 30; i++) {
        const x1 = Math.random() * 16 - 8; // Random x between -8 and 8
        const x2 = Math.random() * 16 - 8; // Random x between -8 and 8
        // Linear function with some noise
        const y = ((0.5 * x1) + (0.3 * x2) + 2 + (Math.random() * 2 - 1)).toFixed(2);
        newPoints.push({ x1, x2, y });
      }
      setTrainingPoints(newPoints);
    }
    
    // Reset other states
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setShowTree(false);
    setTreeImg(null);
    setDecisionBoundaryImg(null);
  };

  // Coordinate transformation functions
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
    const dataX = scale.x.min + (adjustedX / canvas.width) * (scale.x.max - scale.x.min);
    const dataY = scale.y.max - (adjustedY / canvas.height) * (scale.y.max - scale.y.min);
    
    return { x: dataX, y: dataY };
  };

  // Handle canvas click for classification
  const handleClassificationCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || loading) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    if (pointsMode === 'train') {
      // Add point to training data
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y,
        y: selectedClass
      };
      
      setTrainingPoints([...trainingPoints, newPoint]);
      
      // Hide tree visualization if it was showing
      if (showTree) {
        setShowTree(false);
        setTreeImg(null);
        setDecisionBoundaryImg(null);
      }
    } else {
      // Add point to predict list
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y
      };
      
      setPredictPoints([...predictPoints, newPoint]);
    }
  };
  
  // Handle canvas click for regression
  const handleRegressionCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || loading) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    if (pointsMode === 'train') {
      // Store temporary point and show dialog
      setTempPoint({
        x1: dataPoint.x,
        x2: dataPoint.y
      });
      
      // Position dialog near click but ensure it stays on screen
      const dialogWidth = 200;
      const dialogHeight = 120;
      let dialogX = e.clientX - (dialogWidth / 2);
      let dialogY = e.clientY - dialogHeight - 10; // position above cursor
      
      // Adjust if off screen
      if (dialogX < 10) dialogX = 10;
      if (dialogX > window.innerWidth - dialogWidth - 10) dialogX = window.innerWidth - dialogWidth - 10;
      if (dialogY < 10) dialogY = e.clientY + 10; // position below cursor instead
      
      setDialogPosition({ x: dialogX, y: dialogY });
      setTempValue('');
      setShowPointDialog(true);
      
      // Hide tree visualization if it was showing
      if (showTree) {
        setShowTree(false);
        setTreeImg(null);
        setDecisionBoundaryImg(null);
      }
    } else {
      // Add point to predict list
      const newPoint = {
        x1: dataPoint.x,
        x2: dataPoint.y
      };
      
      setPredictPoints([...predictPoints, newPoint]);
    }
  };
  
  // Handle canvas mouse move for hover effect
  const handleCanvasMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const dataPoint = screenToData(x, y);
    
    // Find if mouse is near any prediction point (within a small radius)
    const hoverRadius = 0.5; // Data units
    let foundPoint = null;
    
    // Check all prediction points
    for (const point of predictions.length > 0 ? predictions : predictPoints) {
      const dist = Math.sqrt(
        Math.pow(point.x1 - dataPoint.x, 2) + 
        Math.pow(point.x2 - dataPoint.y, 2)
      );
      
      if (dist < hoverRadius) {
        foundPoint = point;
        break;
      }
    }
    
    if (foundPoint !== hoveredPoint) {
      setHoveredPoint(foundPoint);
    }
  };
  
  // Handle canvas mouse leave
  const handleCanvasMouseLeave = () => {
    setHoveredPoint(null);
  };
  
  // Handle dialog confirmation
  const handleDialogConfirm = () => {
    // Validate input is a number
    const numValue = parseFloat(tempValue);
    
    if (tempPoint && !isNaN(numValue)) {
      // Add point to training data with value
      const newPoint = {
        x1: tempPoint.x1,
        x2: tempPoint.x2,
        y: tempValue
      };
      
      setTrainingPoints([...trainingPoints, newPoint]);
      setShowPointDialog(false);
      setTempPoint(null);
      setError(null); // Clear any errors
    } else {
      // Show error if not a valid number
      setError('Please enter a valid number for the value');
      // Keep dialog open
    }
  };
  
  // Handle canvas click based on active tab
  const handleCanvasClick = (e) => {
    if (animatingPrediction) return; // Prevent clicks during animation
    
    if (activeTab === 'classification') {
      handleClassificationCanvasClick(e);
    } else {
      handleRegressionCanvasClick(e);
    }
  };

  // Reset the visualization
  const resetData = () => {
    setTrainingPoints([]);
    setPredictPoints([]);
    setPredictions([]);
    setPointsMode('train');
    setError(null);
    setShowTree(false);
    setTreeImg(null);
    setDecisionBoundaryImg(null);
    setAnimatingPrediction(false);
    setCurrentAnimationPath([]);
    setCurrentAnimationStep(0);
    setCurrentHighlightedNode(null);
    setHoveredPoint(null);
  };
  
// Update the trainModel function to include more logging and properly handle images
const trainModel = () => {
  if (trainingPoints.length < 2) {
    setError("Need at least 2 training points");
    return;
  }
  
  setLoading(true);
  setError(null);
  
  // Add debugging - log the raw training points
  console.log("Raw training points:", JSON.stringify(trainingPoints, null, 2));
  
  // Format the training points properly - with strict type conversion
  const formattedPoints = trainingPoints.map(point => {
    try {
      // Handle potential string/undefined values strictly
      const x1 = Number(point.x1);
      const x2 = Number(point.x2);
      
      if (isNaN(x1) || isNaN(x2)) {
        console.error("Invalid coordinate values:", point);
        throw new Error("Invalid coordinate values");
      }
      
      if (activeTab === 'classification') {
        return {
          x1: x1,
          x2: x2,
          class: String(point.y || '') // Ensure class is a string, default to empty if undefined
        };
      } else {
        const yValue = Number(point.y);
        if (isNaN(yValue)) {
          console.error("Invalid y value for regression:", point);
          throw new Error("Invalid y value for regression");
        }
        
        return {
          x1: x1,
          x2: x2,
          value: yValue
        };
      }
    } catch (err) {
      console.error("Error formatting point:", point, err);
      throw err;
    }
  });
  
  // Add more debugging - log the formatted points
  console.log("Formatted points:", JSON.stringify(formattedPoints, null, 2));
  
  const requestData = {
    trained_points: formattedPoints,
    type: activeTab,
    parameters: {
      max_depth: maxDepth,
      min_samples_split: minSamplesSplit,
      criterion: criterion
    }
  };
  
  console.log("Sending request:", JSON.stringify(requestData, null, 2));
  
  axios.post(`${apiUrl}/dtree/train`, requestData)
    .then(response => {
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      // Show tree visualization
      setShowTree(true);
      
      // Check if tree_visualization exists and is not too large
      if (response.data.tree_visualization) {
        setTreeImg(response.data.tree_visualization);
        console.log("Tree visualization received, length:", 
                   response.data.tree_visualization.length);
      } else {
        console.warn("No tree visualization in response");
        setError("Tree visualization not available");
      }
      
      // Also show decision boundary
      if (response.data.decision_boundary) {
        setDecisionBoundaryImg(response.data.decision_boundary);
        console.log("Decision boundary received, length:",
                   response.data.decision_boundary.length);
      } else {
        console.warn("No decision boundary in response");
      }
      
      console.log("Training successful - tree and boundary images received");
    })
    .catch(err => {
      console.error("Error training model:", err);
      
      // Show more detailed error information
      if (err.response && err.response.data) {
        console.error("Backend error data:", err.response.data);
      }
      
      // Show the full error message from the backend if available
      const errorMsg = err.response?.data?.error || err.message;
      setError("Failed to train model: " + errorMsg);
    })
    .finally(() => {
      setLoading(false);
    });
};
  
// Update the handlePrediction function to better handle results
  const handlePrediction = () => {
    if (predictPoints.length === 0) {
      setError("Add some points to predict first");
      return;
    }
    
    if (!showTree) {
      setError("Train the model first");
      return;
    }
    
    setLoading(true);
    setError(null);
    setAnimatingPrediction(true);
    
    // Format the points properly
    const formattedPoints = predictPoints.map(point => ({
      x1: parseFloat(point.x1),
      x2: parseFloat(point.x2)
    }));
    
    const requestData = {
      trained_points: trainingPoints,
      predict_points: formattedPoints,
      type: activeTab,
      parameters: {
        max_depth: maxDepth,
        min_samples_split: minSamplesSplit,
        criterion: criterion
      }
    };
    
    console.log("Sending predict request:", JSON.stringify(requestData, null, 2));
    
    // In the handlePrediction function
    axios.post(`${apiUrl}/dtree/predict`, requestData)
    .then(response => {
      if (response.data.error) {
        setError(response.data.error);
        setAnimatingPrediction(false);
        setLoading(false);
        return;
      }
      
      // Add debug logging
      console.log("Prediction response:", response.data);
      
      // Get prediction results and ensure they have the correct format
      const results = response.data.predictions || [];
      const classMapping = response.data.class_mapping || {};
      
      console.log("Raw predictions:", results);
      console.log("Class mapping:", classMapping);
      
      // Format the prediction results to match with predict points
      const formattedResults = results.map((result, index) => {
        const point = predictPoints[index];
        return {
          x1: point.x1,
          x2: point.x2,
          predictedClass: result,
          confidence: null
        };
      });
      
      console.log("Formatted results:", formattedResults);
      
      // Update predictions state
      setPredictions(formattedResults);
      setAnimatingPrediction(false);
      setLoading(false);
    })
    .catch(err => {
      console.error("Error making prediction:", err);
      setError("Failed to make prediction: " + (err.response?.data?.error || err.message));
      setAnimatingPrediction(false);
      setLoading(false);
    });
  };
  

  // Canvas drawing effect
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Draw canvas
    const drawCanvas = () => {
      // Get actual canvas dimensions
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      
      // Clear the canvas
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      
      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 0.5;
      
      // Calculate step size for grid lines (16 divisions for -8 to 8 range = 1 unit per division)
      const stepX = canvasWidth / 16;
      const stepY = canvasHeight / 16;
      
      // Draw horizontal grid lines
      for (let i = 0; i <= 16; i++) {
        const y = i * stepY;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      }
      
      // Draw vertical grid lines
      for (let i = 0; i <= 16; i++) {
        const x = i * stepX;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
      }
      
      // Draw X and Y axes with dotted lines
      ctx.strokeStyle = '#9ca3af'; // Medium gray color
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); // Dotted line pattern
      
      // X-axis (horizontal line at y=0)
      const yAxisPos = canvasHeight / 2; // y=0 position
      ctx.beginPath();
      ctx.moveTo(0, yAxisPos);
      ctx.lineTo(canvasWidth, yAxisPos);
      ctx.stroke();
      
      // Y-axis (vertical line at x=0)
      const xAxisPos = canvasWidth / 2; // x=0 position
      ctx.beginPath();
      ctx.moveTo(xAxisPos, 0);
      ctx.lineTo(xAxisPos, canvasHeight);
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
        ctx.fillText(value.toFixed(0), x - 8, canvasHeight - 5);
      }
      
      // Y-axis labels
      for (let i = 0; i <= 16; i += 2) {
        const y = i * stepY;
        const value = scale.y.max - (i / 16) * (scale.y.max - scale.y.min);
        ctx.fillText(value.toFixed(0), 5, y + 4);
      }
      
      // Draw training points
      trainingPoints.forEach(point => {
        // Map from data coordinates to canvas coordinates
        const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        
        if (activeTab === 'classification') {
          // Different color based on class for classification - compare as string
          const pointClass = point.y.toString();
          
          if (pointClass === '0') {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.7)'; // Blue
          } else if (pointClass === '1') {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.7)'; // Red
          } else if (pointClass === '2') {
            ctx.fillStyle = 'rgba(34, 197, 94, 0.7)'; // Green
          } else {
            ctx.fillStyle = 'rgba(156, 163, 175, 0.7)'; // Gray fallback
          }
        } else {
          // For regression, use a gradient based on value
          // Normalize value to be between 0 and 1 for color gradient
          const normalizedValue = Math.min(Math.max(parseFloat(point.y) / 10, 0), 1);
          const r = Math.round(59 + (239 - 59) * normalizedValue);
          const g = Math.round(130 + (68 - 130) * normalizedValue);
          const b = Math.round(246 + (68 - 246) * normalizedValue);
          
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
          
          // Add text label for the value
          ctx.fillStyle = '#000000';
          ctx.font = '10px Arial';
          ctx.fillText(point.y, x + 10, y - 10);
          
          // Reset fill style for the point
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
        
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      
      // Draw points to predict (before prediction) with smaller radius
      if (predictions.length === 0) {
        predictPoints.forEach(point => {
          // Correct position calculation
          const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
          const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
          
          const isHovered = hoveredPoint && hoveredPoint.x1 === point.x1 && hoveredPoint.x2 === point.x2;
          
          ctx.beginPath();
          ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2); // Bigger if hovered
          
          // Gray color for unpredicted points
          ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)';
          
          ctx.fill();
          ctx.strokeStyle = isHovered ? '#000' : '#333';
          ctx.lineWidth = isHovered ? 2 : 1;
          ctx.stroke();
        });
      }
      
// In the drawCanvas function
// Update the prediction drawing section to better handle class mappings

// Draw prediction results with smaller radius
predictions.forEach(pred => {
  // Skip invalid prediction points
  if (!pred || pred.x1 === undefined || pred.x2 === undefined || pred.predictedClass === undefined) {
    console.warn("Skipping invalid prediction point:", pred);
    return; // Skip this iteration
  }
  
  // Correct position calculation
  const x = ((pred.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
  const y = ((scale.y.max - pred.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
  
  const isHovered = hoveredPoint && hoveredPoint.x1 === pred.x1 && hoveredPoint.x2 === pred.x2;
  
  ctx.beginPath();
  ctx.arc(x, y, isHovered ? 8 : 6, 0, Math.PI * 2); // Bigger if hovered
  
  if (activeTab === 'classification') {
    // Add more logging for debugging
    console.log("Predicted class:", pred.predictedClass, typeof pred.predictedClass);
    
    // Use strict string comparison for class values
    const predClass = String(pred.predictedClass).trim();
    
    // Match exact strings
    if (predClass === '0') {
      ctx.fillStyle = isHovered ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)'; // Blue
    } 
    else if (predClass === '1') {
      ctx.fillStyle = isHovered ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)'; // Red
    }
    else if (predClass === '2') {
      ctx.fillStyle = isHovered ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)'; // Green
    }
    else {
      console.warn("Unknown class value:", predClass);
      ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)'; // Gray
    }
    
    // Add text label for the predicted class
    ctx.fillStyle = '#000000';
    ctx.font = '10px Arial';
    ctx.fillText(String(pred.predictedClass), x + 10, y - 10);
    
    // Reset fill style for the point
    if (predClass === '0') {
      ctx.fillStyle = isHovered ? 'rgba(30, 64, 175, 0.9)' : 'rgba(59, 130, 246, 0.7)'; // Blue
    } 
    else if (predClass === '1') {
      ctx.fillStyle = isHovered ? 'rgba(185, 28, 28, 0.9)' : 'rgba(239, 68, 68, 0.7)'; // Red
    }
    else if (predClass === '2') {
      ctx.fillStyle = isHovered ? 'rgba(21, 128, 61, 0.9)' : 'rgba(34, 197, 94, 0.7)'; // Green
    }
    else {
      ctx.fillStyle = isHovered ? 'rgba(107, 114, 128, 0.9)' : 'rgba(156, 163, 175, 0.7)'; // Gray
    }
  } else {
    // Regression case - unchanged
  }
  
  // Dotted border to indicate prediction
  ctx.setLineDash([2, 2]);
  ctx.strokeStyle = isHovered ? '#000' : '#333';
  ctx.lineWidth = isHovered ? 2 : 1.5;
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
});
      
      // Show current predicting point if animating
      if (animatingPrediction && currentAnimationStep < currentAnimationPath.length && predictPoints[predictions.length]) {
        const point = predictPoints[predictions.length];
        const x = ((point.x1 - scale.x.min) / (scale.x.max - scale.x.min)) * canvasWidth;
        const y = ((scale.y.max - point.x2) / (scale.y.max - scale.y.min)) * canvasHeight;
        
        // Highlight the current point being predicted
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(99, 102, 241, 0.7)'; // Indigo
        ctx.fill();
        ctx.strokeStyle = '#4338ca'; // Darker indigo
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Add "Predicting..." label
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('Predicting...', x - 30, y - 15);
      }
    };
    
    // Draw canvas
    drawCanvas();
    
  }, [trainingPoints, predictPoints, predictions, scale, canvasDimensions, activeTab, hoveredPoint, animatingPrediction, currentAnimationStep, currentAnimationPath]);

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
        <h1 className="model-title">Decision Trees</h1>
      </div>

      <p className="model-description">
        Decision Trees are versatile machine learning algorithms that can be used for both classification and regression tasks.
        They split the data based on feature values to create a tree-like structure for decision making.
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
      
      <div className="tabs">
        <div 
          className={`tab ${activeTab === 'classification' ? 'active' : ''}`}
          onClick={() => handleTabChange('classification')}
        >
          Classification
        </div>
        <div 
          className={`tab ${activeTab === 'regression' ? 'active' : ''}`}
          onClick={() => handleTabChange('regression')}
        >
          Regression
        </div>
      </div>

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
            gridRow: '1 / 3',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div className="section-header">
              <h2 className="section-title">Decision Tree {activeTab === 'classification' ? 'Classification' : 'Regression'}</h2>
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
                  disabled={loading}
                >
                  Reset Data
                </button>
              </div>
            </div>
            
            {error && <div className="error-message">{error}</div>}

            <div style={{ marginBottom: '1rem' }}>
              <p style={{ color: '#4b5563', marginBottom: '0.5rem', lineHeight: '1.5' }}>
                Click on the graph below to add data points. Current mode: <strong>{pointsMode === 'train' ? 'Training' : 'Prediction'}</strong>
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
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  style={{ 
                    display: 'block', 
                    cursor: loading || animatingPrediction ? 'wait' : 'crosshair',
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
                {animatingPrediction ? 'Predicting...' : 
                  (pointsMode === 'train' ? 'Click to add training point' : 'Click to add prediction point')}
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
                  <strong>Training Points:</strong> {trainingPoints.length}
                </div>
                <div>
                  <strong>Points to Predict:</strong> {predictPoints.length}
                </div>
                <div>
                  <strong>Predictions Made:</strong> {predictions.length}
                </div>
              </div>
            </div>
            
            {/* Legend and How Decision Trees Work */}
            <div style={{ 
              width: '100%',
              backgroundColor: 'white', 
              padding: '1rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              fontSize: '0.85rem',
              marginBottom: '1rem'
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
                    borderRadius: '50%', 
                    backgroundColor: 'rgba(34, 197, 94, 0.7)', 
                    border: '1px solid #333' 
                  }}></div>
                  <span>Class 2</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'rgba(156, 163, 175, 0.7)',
                    border: '1px solid #333'
                  }}></div>
                  <span>Unpredicted</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '12px', 
                    height: '12px', 
                    borderRadius: '50%',
                    backgroundColor: 'white',
                    border: '2px dashed #333'
                  }}></div>
                  <span>Predicted</span>
                </div>
              </div>
            </div>
            
            {/* How Decision Trees Work */}
            <div style={{ 
              width: '100%',
              backgroundColor: 'white', 
              padding: '1rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              fontSize: '0.85rem'
            }}>
              <h3 style={{ marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: '500' }}>How Decision Trees Work</h3>
              <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                <p style={{ marginBottom: '0.5rem' }}>
                  Decision Trees work by recursively splitting data:
                </p>
                <ol style={{ paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                  <li>Choose the best feature to split data</li>
                  <li>Split data based on that feature</li>
                  <li>Continue recursively until stopping criteria</li>
                  <li>Make predictions based on majority class/average value in leaf nodes</li>
                </ol>
                <p style={{ fontSize: '0.8rem', fontStyle: 'italic' }}>
                  Controlling tree depth prevents overfitting and improves generalization.
                </p>
              </div>
            </div>
          </div>
          
          {/* Right column: Controls, tree visualization and info boxes */}
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
              
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="depth-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '500', 
                  color: '#4b5563',
                  fontSize: '1rem'
                }}>
                  Maximum Tree Depth: {maxDepth}
                </label>
                <input
                  id="depth-slider"
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>1 (Simple tree)</span>
                  <span>10 (Complex tree)</span>
                </div>
              </div>
              
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="samples-slider" style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '500', 
                  color: '#4b5563',
                  fontSize: '1rem'
                }}>
                  Min Samples to Split: {minSamplesSplit}
                </label>
                <input
                  id="samples-slider"
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={minSamplesSplit}
                  onChange={(e) => setMinSamplesSplit(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  fontSize: '0.85rem', 
                  color: '#6b7280',
                  marginTop: '0.5rem'
                }}>
                  <span>2 (More splits)</span>
                  <span>10 (Fewer splits)</span>
                </div>
              </div>
              
              {activeTab === 'classification' && (
                <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '0.75rem', 
                    fontWeight: '500', 
                    color: '#4b5563',
                    fontSize: '1rem'
                  }}>
                    Split Criterion
                  </label>
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button
                      onClick={() => setCriterion('gini')}
                      style={{
                        padding: '0.75rem 1rem',
                        backgroundColor: criterion === 'gini' ? '#3b82f6' : '#e5e7eb',
                        color: criterion === 'gini' ? 'white' : '#4b5563',
                        border: 'none',
                        borderRadius: '0.5rem',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem'
                      }}
                    >
                      Gini
                    </button>
                    <button
                      onClick={() => setCriterion('entropy')}
                      style={{
                        padding: '0.75rem 1rem',
                        backgroundColor: criterion === 'entropy' ? '#3b82f6' : '#e5e7eb',
                        color: criterion === 'entropy' ? 'white' : '#4b5563',
                        border: 'none',
                        borderRadius: '0.5rem',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem'
                      }}
                    >
                      Entropy
                    </button>
                  </div>
                </div>
              )}
              
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>
                Interaction Mode
              </h3>
              
              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                <button
                  onClick={() => setPointsMode('train')}
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: pointsMode === 'train' ? '#3b82f6' : '#e5e7eb',
                    color: pointsMode === 'train' ? 'white' : '#4b5563',
                    border: 'none',
                    borderRadius: '0.5rem',
                    cursor: loading || animatingPrediction ? 'not-allowed' : 'pointer',
                    fontWeight: '500',
                    flex: 1,
                    fontSize: '0.95rem',
                    opacity: loading || animatingPrediction ? 0.7 : 1
                  }}
                  disabled={loading || animatingPrediction}
                >
                  Training Points
                </button>
                <button
                  onClick={() => setPointsMode('predict')}
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: pointsMode === 'predict' ? '#3b82f6' : '#e5e7eb',
                    color: pointsMode === 'predict' ? 'white' : '#4b5563',
                    border: 'none',
                    borderRadius: '0.5rem',
                    cursor: loading || animatingPrediction ? 'not-allowed' : 'pointer',
                    fontWeight: '500',
                    flex: 1,
                    fontSize: '0.95rem',
                    opacity: loading || animatingPrediction ? 0.7 : 1
                  }}
                  disabled={loading || animatingPrediction}
                >
                  Prediction Points
                </button>
              </div>
              
              {pointsMode === 'train' && activeTab === 'classification' && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <p style={{ marginBottom: '0.75rem', color: '#4b5563', fontSize: '1rem' }}>
                    Select class for training points:
                  </p>
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button
                      onClick={() => setSelectedClass('0')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '0' ? 'rgba(59, 130, 246, 1)' : 'rgba(59, 130, 246, 0.1)',
                        color: selectedClass === '0' ? 'white' : '#1e40af',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem',
                        opacity: loading || animatingPrediction ? 0.7 : 1
                      }}
                      disabled={loading || animatingPrediction}
                    >
                      Class 0
                    </button>
                    <button
                      onClick={() => setSelectedClass('1')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '1' ? 'rgba(239, 68, 68, 1)' : 'rgba(239, 68, 68, 0.1)',
                        color: selectedClass === '1' ? 'white' : '#b91c1c',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem',
                        opacity: loading || animatingPrediction ? 0.7 : 1
                      }}
                      disabled={loading || animatingPrediction}
                    >
                      Class 1
                    </button>
                    <button
                      onClick={() => setSelectedClass('2')}
                      style={{
                        padding: '0.75rem 0.5rem',
                        border: 'none',
                        borderRadius: '0.5rem',
                        backgroundColor: selectedClass === '2' ? 'rgba(34, 197, 94, 1)' : 'rgba(34, 197, 94, 0.1)',
                        color: selectedClass === '2' ? 'white' : '#15803d',
                        cursor: 'pointer',
                        fontWeight: '500',
                        flex: 1,
                        fontSize: '0.95rem',
                        opacity: loading || animatingPrediction ? 0.7 : 1
                      }}
                      disabled={loading || animatingPrediction}
                    >
                      Class 2
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            {/* Action buttons */}
            <div style={{ 
              marginBottom: '1rem',
              backgroundColor: 'white', 
              padding: '1rem', 
              borderRadius: '6px', 
              border: '1px solid #e5e7eb',
              width: '100%'
            }}>
              <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', fontWeight: '500' }}>Actions</h3>
              
              <button 
                onClick={trainModel}
                disabled={loading || backendStatus === "disconnected" || trainingPoints.length < 2 || animatingPrediction}
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
                  opacity: (loading || backendStatus === "disconnected" || trainingPoints.length < 2 || animatingPrediction) ? 0.7 : 1,
                  marginBottom: '1.0rem'
                }}
              >
                {loading ? 'Training...' : 'Train Decision Tree'}
              </button>
              
              <button 
                onClick={handlePrediction}
                disabled={loading || backendStatus === "disconnected" || 
                        trainingPoints.length < 2 || predictPoints.length < 1 || 
                        !showTree || animatingPrediction}
                style={{
                  width: '100%',
                  backgroundColor: loading || animatingPrediction ? '#93c5fd' : '#3b82f6',
                  color: 'white',
                  padding: '0.9rem',
                  fontSize: '1.05rem',
                  fontWeight: '500',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loading || animatingPrediction ? 'wait' : 'pointer',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                  opacity: (loading || backendStatus === "disconnected" || 
                          trainingPoints.length < 2 || predictPoints.length < 1 || 
                          !showTree || animatingPrediction) ? 0.7 : 1,
                  marginBottom: '1.25rem'
                }}
              >
                {loading ? 'Predicting...' : (animatingPrediction ? 'Animating Prediction...' : 'Predict Points')}
              </button>

            </div>
            {showTree && decisionBoundaryImg && (
              <div style={{ 
                marginTop: '1rem',
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '8px', 
                border: '1px solid #e5e7eb',
                maxWidth: '100%',
                overflow: 'hidden'
              }}>
                <h3 style={{ 
                  marginBottom: '0.5rem', 
                  fontSize: '1rem', 
                  fontWeight: '500', 
                  textAlign: 'center' 
                }}>
                  {activeTab === 'classification' ? 'Decision Boundary' : 'Regression Surface'}
                </h3>
                <div style={{ 
                  width: '100%',
                  maxHeight: '400px',
                  display: 'flex',
                  justifyContent: 'center',
                  overflow: 'auto'
                }}>
                  {/* Check if the string already includes data:image prefix */}
                  <img 
                    src={decisionBoundaryImg.startsWith('data:image') ? decisionBoundaryImg : `data:image/png;base64,${decisionBoundaryImg}`}
                    alt={activeTab === 'classification' ? 'Decision Boundary' : 'Regression Surface'}
                    style={{ 
                      maxWidth: '100%',
                      height: 'auto'
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
          {showTree && treeImg && (
              <div style={{ 
                marginTop: '0.5rem',
                backgroundColor: 'white', 
                padding: '1rem', 
                borderRadius: '8px', 
                border: '1px solid #e5e7eb',
                width: '100%',
                overflow: 'hidden' 
              }}>
                <h3 style={{ 
                  marginBottom: '0.5rem', 
                  fontSize: '1rem', 
                  fontWeight: '500', 
                  textAlign: 'center' 
                }}>
                  Decision Tree Visualization (max_depth={maxDepth})
                </h3>
                <div style={{ 
                  width: '100%',
                  maxWidth: '100%',  
                  margin: '0 auto',
                  position: 'relative',
                  paddingBottom: isTreeEnlarged ? '100%' : '70%', 
                  border: '1px solid #e5e7eb',
                  borderRadius: '4px',
                  overflow: 'auto',
                  marginBottom: '0.5rem'
                }}>
                  <div style={{
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    width: '100%', 
                    height: '100%', 
                    overflow: 'auto'
                  }}>
                    <img 
                      src={treeImg.startsWith('data:image') ? treeImg : `data:image/png;base64,${treeImg}`}
                      alt="Decision Tree Visualization"
                      onClick={toggleTreeEnlargement}
                      style={{ 
                        width: isTreeEnlarged ? 'auto' : '100%',
                        maxWidth: isTreeEnlarged ? 'none' : '100%',
                        height: 'auto',
                        objectFit: isTreeEnlarged ? 'contain' : 'contain',
                        cursor: 'zoom-in',
                        transform: isTreeEnlarged ? 'scale(1.5)' : 'scale(1)',
                        transformOrigin: 'top left',
                        transition: 'transform 0.3s ease'
                      }}
                    />
                    
                    
                    {/* Highlight current node in the path if animating */}
                    {animatingPrediction && currentHighlightedNode !== null && (
                      <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        // Note: The actual highlight would be done with SVG manipulation
                        // This is a placeholder that would be replaced by backend SVG coordinates
                        pointerEvents: 'none'
                      }}>
                        <div style={{
                          position: 'absolute',
                          top: '20px', // This would come from the backend
                          right: '20px',
                          backgroundColor: 'rgba(59, 130, 246, 0.5)',
                          padding: '10px',
                          borderRadius: '4px',
                          color: 'white',
                          fontWeight: 'bold'
                        }}>
                          Current node: {currentHighlightedNode}
                        </div>
                      </div>
                    )}
                    {/* Add zoom instructions */}
                    <div style={{
                      position: 'absolute',
                      bottom: '10px',
                      right: '10px',
                      padding: '4px 8px',
                      backgroundColor: 'rgba(255, 255, 255, 0.8)',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      color: '#4b5563',
                      pointerEvents: 'none'
                    }}>
                      {isTreeEnlarged ? 'Click to reduce' : 'Click to enlarge'}
                    </div>
                  </div>
                </div>
                <p style={{ 
                  color: '#6b7280', 
                  fontSize: '0.8rem', 
                  maxWidth: '500px', 
                  margin: '0 auto', 
                  textAlign: 'center',
                  lineHeight: '1.4'
                }}>
                  {activeTab === 'classification' 
                    ? 'The decision tree shows how the model splits data to make classification decisions.'
                    : 'The decision tree shows how the model splits data to predict continuous values.'}
                </p>
                
                {animatingPrediction && (
                  <div style={{ 
                    marginTop: '0.5rem',
                    backgroundColor: '#f0f9ff',
                    padding: '0.5rem',
                    borderRadius: '0.25rem',
                    border: '1px solid #bae6fd',
                    fontSize: '0.8rem',
                    textAlign: 'center'
                  }}>
                    <p style={{ margin: 0, color: '#0284c7' }}>
                      Following decision path for point #{predictions.length + 1}: Step {currentAnimationStep + 1} of {currentAnimationPath.length}
                    </p>
                  </div>
                )}
              </div>
            )}
      </div>
      
            {/* Point Value Dialog */}
            {showPointDialog && (
        <div
          style={{
            position: 'fixed',
            left: dialogPosition.x,
            top: dialogPosition.y,
            zIndex: 1000,
            backgroundColor: 'white',
            padding: '1rem',
            borderRadius: '0.5rem',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)',
            width: '200px'
          }}
        >
          <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.9rem' }}>Enter Y value:</h4>
          <input
            type="number"
            step="0.01"
            value={tempValue}
            onChange={(e) => setTempValue(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #e5e7eb',
              borderRadius: '0.25rem',
              marginBottom: '0.75rem'
            }}
            autoFocus
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleDialogConfirm();
              }
            }}
          />
          <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setShowPointDialog(false)}
              style={{
                padding: '0.4rem 0.75rem',
                border: 'none',
                borderRadius: '0.25rem',
                backgroundColor: '#f3f4f6',
                color: '#4b5563',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleDialogConfirm}
              style={{
                padding: '0.4rem 0.75rem',
                border: 'none',
                borderRadius: '0.25rem',
                backgroundColor: '#3b82f6',
                color: 'white',
                cursor: 'pointer'
              }}
            >
              Add Point
            </button>
          </div>
        </div>
      )}
      
      {showSampleDataModal && (
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
                Generate Sample Data ({activeTab === 'classification' ? 'Classification' : 'Regression'})
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
            
            {activeTab === 'classification' ? (
              <>
                {/* Classification options */}
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
                      min="1"
                      max="3"
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
                      <span>1 clusters</span>
                      <span>3 clusters</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Regression options */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <label style={{ 
                    display: 'block', 
                    marginBottom: '0.5rem', 
                    fontWeight: '500', 
                    color: '#4b5563' 
                  }}>
                    Regression Function Type
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                    <button
                      onClick={() => setSampleDataType('linear')}
                      style={{
                        padding: '0.5rem 0.75rem',
                        backgroundColor: sampleDataType === 'linear' ? '#3b82f6' : '#e5e7eb',
                        color: sampleDataType === 'linear' ? 'white' : '#4b5563',
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
                      <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Simple linear pattern</span>
                    </button>
                    <button
                      onClick={() => setSampleDataType('nonlinear')}
                      style={{
                        padding: '0.5rem 0.75rem',
                        backgroundColor: sampleDataType === 'nonlinear' ? '#3b82f6' : '#e5e7eb',
                        color: sampleDataType === 'nonlinear' ? 'white' : '#4b5563',
                        border: 'none',
                        borderRadius: '0.5rem',
                        cursor: 'pointer',
                        fontSize: '0.9rem',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center'
                      }}
                    >
                      <span style={{ fontWeight: '500' }}>Nonlinear</span>
                      <span style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Complex regions</span>
                    </button>
                  </div>
                </div>
                
                {/* Sample count slider for regression */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <label htmlFor="reg-sample-count" style={{ 
                    display: 'block', 
                    marginBottom: '0.5rem', 
                    fontWeight: '500', 
                    color: '#4b5563' 
                  }}>
                    Number of Samples: {sampleCount}
                  </label>
                  <input
                    id="reg-sample-count"
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
                
                {/* Noise slider for regression */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <label htmlFor="reg-variance" style={{ 
                    display: 'block', 
                    marginBottom: '0.5rem', 
                    fontWeight: '500', 
                    color: '#4b5563' 
                  }}>
                    Noise Level: {sampleVariance.toFixed(1)}
                  </label>
                  <input
                    id="reg-variance"
                    type="range"
                    min="0.1"
                    max="1.5"
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
                    <span>0.1 (Low noise)</span>
                    <span>1.5 (High noise)</span>
                  </div>
                </div>
                <div style={{ marginBottom: '1.25rem' }}>
                  <label htmlFor="reg-sparsity" style={{ 
                    display: 'block', 
                    marginBottom: '0.5rem', 
                    fontWeight: '500', 
                    color: '#4b5563' 
                  }}>
                    Data Sparsity: {sampleSparsity.toFixed(1)}
                  </label>
                  <input
                    id="reg-sparsity"
                    type="range"
                    min="0.5"
                    max="3.0"
                    step="0.1"
                    value={sampleSparsity}
                    onChange={(e) => setSampleSparsity(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    fontSize: '0.8rem', 
                    color: '#6b7280',
                    marginTop: '0.25rem'
                  }}>
                    <span>0.5 (Dense points)</span>
                    <span>3.0 (Sparse clusters)</span>
                  </div>
                </div>
              </>
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
      )}
    </motion.div>
  );
}

export default DTrees;
        
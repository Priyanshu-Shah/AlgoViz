import axios from 'axios';
// Make sure this URL matches your backend server address and port
const API_URL = 'http://localhost:5000/api';

export const getModels = async () => {
  const response = await axios.get(`${API_URL}`);
  return response.data;
};

export const getModelDetails = async (id) => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

export const runLinearRegression = async (data) => {
  try {
    console.log('API call to:', `${API_URL}/linear-regression`);
    console.log('Data being sent:', JSON.stringify(data, null, 2));
    
    // Add timeout to prevent hanging requests
    const response = await axios.post(`${API_URL}/linear-regression`, data, {
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    
    console.log('API raw response status:', response.status);
    console.log('API raw response type:', typeof response.data);
    
    // Directly check if response.data exists, has expected properties
    if (!response.data) {
      console.error('Empty response received from server');
      return { error: 'Empty response received from server' };
    }
    
    // Additional check for basic result structure
    if (response.data && 
        typeof response.data === 'object' && 
        !response.data.error) {
      const keys = Object.keys(response.data);
      console.log('Response contains keys:', keys);
      
      // Ensure all numeric values are converted to numbers
      const processedData = {...response.data};
      for (const key in processedData) {
        if (key !== 'plot' && key !== 'equation' && processedData[key] !== null) {
          const num = parseFloat(processedData[key]);
          if (!isNaN(num)) {
            processedData[key] = num;
          }
        }
      }
      
      // Map the train metrics to the expected names in the frontend
      if (processedData.mse_train !== undefined && processedData.r2_train !== undefined) {
        processedData.mse = processedData.mse_train;
        processedData.r2 = processedData.r2_train;
      }

      // Remove the test metrics if you don't need them
      delete processedData.mse_test;
      delete processedData.r2_test;
      delete processedData.mse_train;
      delete processedData.r2_train;
      
      return processedData;
    }
    
    // Return the data as is (might contain error info)
    return response.data;
  } catch (error) {
    console.error('API error in runLinearRegression:', error);
    
    let errorMessage = 'Network error or server unavailable';
    let errorDetails = {};
    
    if (error.response) {
      // The server responded with a status code outside the 2xx range
      console.error('Server error status:', error.response.status);
      console.error('Server error data:', error.response.data);
      errorMessage = error.response.data.error || `Server error: ${error.response.status}`;
      errorDetails = error.response.data;
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
      errorMessage = 'No response received from server';
    } else {
      // Something happened in setting up the request
      console.error('Request setup error:', error.message);
      errorMessage = `Request error: ${error.message}`;
    }
    
    return {
      error: errorMessage,
      details: errorDetails,
      isClientError: true
    };
  }
};

export const getLinearRegressionSampleData = async (n_samples = 30, noise = 5.0) => {
  const response = await axios.get(
    `${API_URL}/linear-regression/sample?n_samples=${n_samples}&noise=${noise}`
  );
  return response.data;
};

export const getSVMSampleData = async (n_samples = 30, noise = 0.1) => {
  const response = await axios.get(
    `${API_URL}/svm/sample?n_samples=${n_samples}&noise=${noise}`
  );
  return response.data;
};

export const runSVM = async (data) => {
  const response = await axios.post(`${API_URL}/svm`, data);
  return response.data;
}

export const runKnnClassification = async (data) => {
  const response = await axios.post(`${API_URL}/knn-classification`, data);
  return response.data;
};

export const runKnnRegression = async (data) => {
  const response = await axios.post(`${API_URL}/knn-regression`, data);
  return response.data;
};

export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  } catch (error) {
    return { status: 'error', message: error.message };
  }
};

export const runPCA = async (data) => {
  try {
    // Fix the URL: remove the duplicate /api/ path
    const response = await axios.post(`${API_URL}/pca`, data);
    return response.data;
  } catch (error) {
    console.error('Error running PCA:', error);
    throw error.response?.data || error;
  }
};

export const getPCASampleData = async (n_samples = 30, noise = 5.0) => {
  try {
    const response = await axios.get(
      `${API_URL}/pca/sample-data?n_samples=${n_samples}&noise=${noise}`
    );
    return response.data;
  } catch (error) {
    console.error('Error getting PCA sample data:', error);
    throw error.response?.data || error;
  }
};

export const runDBSCAN = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/DBScan`, data);
    return response.data;
  } catch (error) {
    console.error('Error running DBSCAN:', error);
    return {
      status: "error",
      message: error.response?.data?.error || error.message
    };
  }
};

// In api.jsx, make sure this function uses the correct HTTP method (POST)
export const getDBSCANSampleData = async (options) => {
  try {
    // Using POST as defined in your server
    const response = await axios.post(`${API_URL}/DBScan/sample-data`, options);
    return response.data;
  } catch (error) {
    console.error('Error getting DBSCAN sample data:', error);
    throw error.response?.data || error;
  }
};
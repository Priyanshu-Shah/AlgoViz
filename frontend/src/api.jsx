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

// Update these functions:

export async function runPolynomialRegression(data) {
  try {
    // Make sure we're using the correct API URL with port 5000
    const API_URL = 'http://localhost:5000/api';
    
    const response = await fetch(`${API_URL}/regression`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error running polynomial regression:', error);
    throw error;
  }
}

export async function getPolynomialRegressionSampleData(count = 30, noise = 5, degree = 2) {
  try {
    // Make sure we're using the correct API URL with port 5000
    const API_URL = 'http://localhost:5000/api';
    
    const response = await fetch(`${API_URL}/regression/sample?count=${count}&noise=${noise}&degree=${degree}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting sample data:', error);
    throw error;
  }
}

export const getSVMSampleData = async (n_samples = 30, noise = 0.1) => {
  const response = await axios.get(
    `${API_URL}/svm/sample?n_samples=${n_samples}&noise=${noise}`
  );
  return response.data;
};

export const runSVM = async (data) => {
  const response = await axios.post(`${API_URL}/svm`, data);
  return response.data;
};

export const runANN = async (data) => {
  const response = await axios.post(`${API_URL}/ann`, data);
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
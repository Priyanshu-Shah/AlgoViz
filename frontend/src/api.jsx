import axios from 'axios';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const getModels = async () => {
  const response = await axios.get(`${API_URL}`);
  return response.data;
};

export const getModelDetails = async (id) => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

// Polynomial Regression
export async function runPolynomialRegression(data) {
  try {
    const response = await axios.post(`${API_URL}/regression`, data);
    return response.data;
  } catch (error) {
    console.error('Error running polynomial regression:', error);
    throw error;
  }
}

export async function getPolynomialRegressionSampleData(dataset_type = 'linear', n_samples = 30, noise_level = 0.5) {
  try {
    const response = await axios.post(`${API_URL}/regression/sample_data`, {
      dataset_type: dataset_type,
      n_samples: n_samples,
      noise_level: noise_level
    });
    return response.data;
  } catch (error) {
    console.error('Error getting sample data:', error);
    throw error;
  }
}

// SVM
export const getSVMSampleData = async (n_samples = 30, noise = 0.1) => {
  try {
    const response = await axios.get(`${API_URL}/svm/sample?n_samples=${n_samples}&noise=${noise}`);
    return response.data;
  } catch (error) {
    console.error('Error getting SVM sample data:', error);
    throw error.response?.data || error;
  }
};

export const runSVM = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/svm`, data);
    return response.data;
  } catch (error) {
    console.error('Error running SVM:', error);
    throw error.response?.data || error;
  }
};

// ANN
export const runANN = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/ann`, data);
    return response.data;
  } catch (error) {
    console.error('Error running ANN:', error);
    throw error.response?.data || error;
  }
};

export const getANNSampleData = async (dataset_type = 'blobs', n_samples = 100, n_clusters = 2, variance = 0.5) => {
  try {
    const response = await axios.get(
      `${API_URL}/ann/sample?dataset_type=${dataset_type}&n_samples=${n_samples}&n_clusters=${n_clusters}&variance=${variance}`
    );
    return response.data;
  } catch (error) {
    console.error('Error getting ANN sample data:', error);
    throw error.response?.data || error;
  }
};

// KNN
export const runKnnClassification = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/knn-classification`, data);
    return response.data;
  } catch (error) {
    console.error('Error running KNN classification:', error);
    throw error.response?.data || error;
  }
};

export const runKnnRegression = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/knn-regression`, data);
    return response.data;
  } catch (error) {
    console.error('Error running KNN regression:', error);
    throw error.response?.data || error;
  }
};

export const predictKnnPoint = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/knn-predict-point`, data);
    return response.data;
  } catch (error) {
    console.error('Error predicting KNN point:', error);
    throw error.response?.data || error;
  }
};

export const getKnnDecisionBoundary = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/knn-decision-boundary`, data);
    return response.data;
  } catch (error) {
    console.error('Error getting KNN decision boundary:', error);
    throw error.response?.data || error;
  }
};

// Decision Trees
export const runDTrees = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/dtrees`, data);
    return response.data;
  } catch (error) {
    console.error('Error running Decision Trees:', error);
    throw error.response?.data || error;
  }
};

export const getDTreesSampleData = async (dataset_type = 'blobs', n_samples = 40, n_clusters = 3, variance = 0.5) => {
  try {
    const response = await axios.get(
      `${API_URL}/dtrees/sample?dataset_type=${dataset_type}&n_samples=${n_samples}&n_clusters=${n_clusters}&variance=${variance}`
    );
    return response.data;
  } catch (error) {
    console.error('Error getting Decision Trees sample data:', error);
    throw error.response?.data || error;
  }
};

// K-Means
export const runKMeans = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/kmeans`, data);
    return response.data;
  } catch (error) {
    console.error('Error running K-Means:', error);
    throw error.response?.data || error;
  }
};

export const getKMeansSampleData = async (dataset_type = 'blobs', n_samples = 100, n_clusters = 3, variance = 0.5) => {
  try {
    const response = await axios.get(
      `${API_URL}/kmeans/sample?dataset_type=${dataset_type}&n_samples=${n_samples}&n_clusters=${n_clusters}&variance=${variance}`
    );
    return response.data;
  } catch (error) {
    console.error('Error getting K-Means sample data:', error);
    throw error.response?.data || error;
  }
};

// PCA
export const runPCA = async (data) => {
  try {
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

// DBSCAN
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

export const getDBSCANSampleData = async (options) => {
  try {
    const response = await axios.post(`${API_URL}/DBScan/sample-data`, options);
    return response.data;
  } catch (error) {
    console.error('Error getting DBSCAN sample data:', error);
    throw error.response?.data || error;
  }
};

// Health check
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  } catch (error) {
    return { status: 'error', message: error.message };
  }
};
import axios from 'axios';
const API_URL = 'http://localhost:5000';

export const getModels = async () => {
  const response = await axios.get(`${API_URL}`);
  return response.data;
};

export const getModelDetails = async (id) => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

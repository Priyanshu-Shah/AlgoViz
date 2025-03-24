# Machine Learning Visualization Project

An interactive web application for exploring and visualizing machine learning models. This project allows users to input data, train models, and see visualizations of the results.

## Features

- Interactive UI with modern animations using Framer Motion
- Categorized machine learning models (supervised and unsupervised)
- Real-time model training and visualization
- Responsive design for all devices
- Sample data generation for quick testing

## Currently Implemented Models

- **Linear Regression**: Predicts continuous values based on linear relationships
- **K-Nearest Neighbors (KNN)**: Classification and regression based on nearest data points

## Project Structure

```
PRML_project/
├── app.py                 # Flask backend entry point
├── models/                # Machine learning model implementations
│   ├── linReg.py          # Linear regression model
│   └── knn.py             # K-Nearest Neighbors model
├── datasets/              # Data generation utilities
│   └── sample_data.py     # Sample data generators
├── frontend/              # React frontend
│   └── src/               # React source code
```

## Setup Instructions

### Simplified Backend Setup

1. Install the required Python packages directly:
   ```bash
   pip install flask flask-cors python-dotenv numpy pandas scikit-learn matplotlib seaborn joblib
   ```

2. Start the Flask backend:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

## Quick Start

Use the simplified start scripts:

- Windows: Double-click `start_backend.bat` then `start_frontend.bat`
- macOS/Linux: Run `./start_backend.sh` then `./start_frontend.sh`

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Select a machine learning model from the homepage
3. Input your data points manually or load sample data
4. Configure model parameters if needed
5. Run the model to see results

## Troubleshooting

If you encounter any issues:

1. Make sure port 5000 is not in use by another application
2. Check the backend health at http://localhost:5000/api/health
3. If you have dependency issues, try installing the packages one by one:
   ```bash
   pip install flask
   pip install flask-cors
   pip install python-dotenv
   # and so on...
   ```

## Future Enhancements

- Additional machine learning models
- Sample datasets
- Model parameter tuning
- Data preprocessing options
- Downloadable results
- Comparative analysis between models
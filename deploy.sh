#!/bin/bash

# Exit on any error
set -e

echo "==== PRML Project Deployment Script ===="
echo "This script will deploy your ML visualizer to Google Cloud Platform"

# Check if gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK (gcloud) is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "Firebase CLI is not installed. Installing it now..."
    npm install -g firebase-tools
fi

# Configuration
echo "Please enter your Google Cloud Project ID:"
read PROJECT_ID

# Set the project
echo "Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Deploy backend to Cloud Run
echo "Deploying backend to Cloud Run..."
cd backend
gcloud builds submit --config=cloudbuild.yaml .

# Get the deployed URL
BACKEND_URL=$(gcloud run services describe prml-backend --platform managed --region us-central1 --format="value(status.url)")
echo "Backend deployed to: $BACKEND_URL"

# Update the frontend environment with the backend URL
cd ../frontend
echo "Updating frontend configuration with backend URL..."
echo "REACT_APP_API_URL=${BACKEND_URL}/api" > .env.production

# Build the React frontend
echo "Building the frontend..."
npm install
npm run build

# Deploy to Firebase
echo "Deploying frontend to Firebase..."
firebase login
firebase use $PROJECT_ID
firebase deploy --only hosting

echo "==== Deployment Complete ===="
echo "Your ML visualizer has been deployed!"
echo "Backend URL: $BACKEND_URL"
echo "Frontend URL: https://$PROJECT_ID.web.app"
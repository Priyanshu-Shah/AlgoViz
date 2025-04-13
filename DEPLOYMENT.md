# Deploying Your ML Visualizer to Google Cloud Platform

This guide will walk you through deploying your PRML Project using your $50 Google Cloud credit.

## Architecture

The deployment follows this architecture:
- **Backend**: Flask API deployed on Google Cloud Run
- **Frontend**: React app deployed on Firebase Hosting
- **Connection**: Frontend makes API calls to the backend

## Prerequisites

1. A Google Cloud Platform account with billing enabled
2. Google Cloud SDK (gcloud) installed locally
3. Firebase CLI installed locally
4. Git (optional, for version control)

## Step-by-Step Deployment

### 1. Initial Setup

#### Set Up Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Make sure billing is enabled and your $50 credit is applied
4. Note your Project ID for later use

#### Install Required Tools
```bash
# Install Google Cloud SDK
# Follow instructions at https://cloud.google.com/sdk/docs/install

# Install Firebase CLI
npm install -g firebase-tools
```

### 2. Deploy the Backend to Cloud Run

1. Navigate to your backend directory:
   ```bash
   cd /path/to/PRML_project/backend
   ```

2. Build and deploy using Cloud Build:
   ```bash
   gcloud builds submit --config=cloudbuild.yaml .
   ```

3. After deployment, note the URL of your Cloud Run service:
   ```bash
   gcloud run services describe prml-backend --platform managed --region us-central1 --format="value(status.url)"
   ```

### 3. Deploy the Frontend to Firebase

1. Initialize a Firebase project (if you haven't already):
   ```bash
   firebase login
   firebase init
   ```
   
   When prompted:
   - Select "Hosting"
   - Select your GCP project
   - Specify "build" as your public directory
   - Configure as a single-page app: Yes

2. Update the production environment file with your backend URL:
   ```bash
   echo "REACT_APP_API_URL=https://your-backend-url.a.run.app/api" > .env.production
   ```

3. Build the React app:
   ```bash
   npm install
   npm run build
   ```

4. Deploy to Firebase:
   ```bash
   firebase deploy --only hosting
   ```

5. Your app will be available at: `https://your-project-id.web.app`

### 4. Using the Deployment Script

For convenience, a deployment script is provided to automate these steps:

```bash
# Make the script executable
chmod +x deploy.sh

# Run the script
./deploy.sh
```

Follow the prompts to complete the deployment.

## Managing Costs

To avoid exceeding your $50 credit:

1. **Set up budget alerts**: In Google Cloud Console, go to Billing > Budgets & Alerts
2. **Use free tier resources**: Cloud Run has a generous free tier
3. **Delete resources** when not in use:
   ```bash
   # Delete Cloud Run service
   gcloud run services delete prml-backend --region us-central1 --platform managed
   ```

## Troubleshooting

1. **CORS Issues**: If you see CORS errors, verify your backend is configured correctly to accept requests from your frontend domain

2. **Cloud Run Error**: If deployment fails:
   - Check quotas in Google Cloud Console
   - Verify Docker build succeeds locally

3. **Firebase Deployment Issues**:
   - Verify you have correct permissions for the project
   - Try running `firebase logout` and then `firebase login` again

## Estimated Costs

With Google Cloud's free tier and your $50 credit:
- Cloud Run: Free for most use cases (first 2 million requests free per month)
- Firebase Hosting: Free for most use cases (first 10GB/month free)
- Cloud Build: First 120 build-minutes/day free

For a small educational project like this, you're unlikely to exceed the free tier limits unless it receives substantial traffic.

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firebase Hosting Documentation](https://firebase.google.com/docs/hosting)
- [Google Cloud Free Tier](https://cloud.google.com/free)
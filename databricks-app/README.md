# Databricks App - Email Generator with React Frontend

This is a migrated version of the email generator app that combines a FastAPI backend with a React frontend, designed to run on Databricks Apps.

## Architecture

### Backend (FastAPI)
- **app.py**: Main FastAPI application that:
  - Serves the email generation API endpoints
  - Integrates with MLflow for experiment tracking
  - Serves the static React frontend files

### Frontend (React)
- Built React app served as static files from the `/static` directory
- Uses relative API paths to communicate with the backend
- Automatically works in both development and production environments

## Deployment

### Prerequisites
1. Databricks CLI configured with your workspace
2. Node.js and npm installed locally
3. A Databricks App already created in your workspace

### Deploy Steps

1. **Set your deployment parameters**:
   ```bash
   # Edit these in deploy.sh or pass as arguments
   APP_FOLDER_IN_WORKSPACE="/Workspace/Users/your.email@company.com/fake_app"
   LAKEHOUSE_APP_NAME="fake-app"
   ```

2. **Run the deployment script**:
   ```bash
   # From the root directory (parent of databricks-app)
   ./deploy.sh [APP_FOLDER_IN_WORKSPACE] [LAKEHOUSE_APP_NAME]
   ```

   The script will:
   - Build the React frontend
   - Copy built files to `databricks-app/static`
   - Upload everything to your Databricks workspace
   - Deploy the app

3. **Access your app**:
   ```
   https://your-workspace.databricks.com/apps/fake-app
   ```

## Environment Variables

The app uses several environment variables configured in `app.yaml`:
- `MLFLOW_EXPERIMENT_ID`: MLflow experiment for tracking
- `LLM_MODEL`: Model name for email generation
- `VITE_DATABRICKS_HOST`: Databricks host for frontend MLflow links
- `VITE_MLFLOW_EXPERIMENT_ID`: Same as MLFLOW_EXPERIMENT_ID for frontend

## Development

### Local Development
1. **Backend**: 
   ```bash
   cd databricks-app
   uvicorn app:app --reload
   ```

2. **Frontend** (in a separate terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Key Changes from Original Structure
1. **Unified deployment**: Both frontend and backend deploy together
2. **Static file serving**: FastAPI serves the built React app
3. **Relative API paths**: Frontend uses `/api/*` paths instead of hardcoded URLs
4. **Environment variable handling**: Configured via app.yaml

## Troubleshooting

### View Logs
```
https://your-app-name.databricksapps.your-workspace.com/logz
```

### Common Issues
1. **404 errors**: Ensure the static directory exists and contains the built frontend
2. **API connection errors**: Check that all `/api/*` routes are defined before the static mount
3. **Build failures**: Try `npm run build:ignore-types` to skip TypeScript errors

## File Structure
```
databricks-app/
├── app.py              # FastAPI backend + static file serving
├── app.yaml            # Databricks Apps configuration
├── requirements.txt    # Python dependencies
├── llm_utils.py        # LLM utility functions
├── input_data.jsonl    # Sample customer data
├── static/             # Built React app (created during deployment)
│   ├── index.html
│   ├── assets/
│   └── ...
└── README.md           # This file
```

## Features

- Generate personalized emails using LLM (via Databricks OpenAI endpoints)
- Stream email generation token by token
- Load and serve customer data
- Submit user feedback with MLflow tracking
- Health checks and environment variable validation

## Required Environment Variables

Make sure to set these environment variables before running the application:

```bash
export DATABRICKS_HOST="your-databricks-host"
export DATABRICKS_TOKEN="your-databricks-token"
export MLFLOW_TRACKING_URI="your-mlflow-tracking-uri"
export MLFLOW_EXPERIMENT_ID="your-experiment-id"
export LLM_MODEL="your-llm-model-name"
```

## Authentication Setup

For testing deployed apps, you'll need to set up Databricks CLI authentication:

1. Configure your Databricks CLI profile:
```bash
databricks configure --profile DEFAULT
# Set auth_type to databricks-cli when prompted
```

2. Authenticate:
```bash
databricks auth login --profile DEFAULT
```

Your `~/.databrickscfg` should look like:
```ini
[DEFAULT]
host      = https://your-workspace.databricks.com
auth_type = databricks-cli
```

## API Endpoints

- `GET /api/hello` - Simple health check
- `GET /api/health` - Detailed health check with OpenAI client status
- `GET /api/env-check` - Check if all environment variables are set
- `GET /api/companies` - Get list of all company names
- `GET /api/customer/{company_name}` - Get customer data by company name
- `POST /api/generate-email/` - Generate an email for a customer
- `POST /api/generate-email-stream/` - Stream email generation token by token
- `POST /api/feedback` - Submit user feedback

## Testing

The test script (`test_migration.py`) uses Databricks SDK authentication to test deployed apps:

- Tests all API endpoints
- Measures response latency
- Tests both regular and streaming email generation
- Validates authentication and connectivity

## Notes

- The application uses MLflow for tracking and logging
- OpenAI client is initialized through Databricks SDK
- CORS is enabled for various frontend URLs including localhost ports and wildcard for Databricks compatibility
- For special MLflow dependencies, the original project uses custom wheels specified in requirements.txt 
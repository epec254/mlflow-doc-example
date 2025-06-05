# Personalized Email Generation App

A FastAPI + React application that generates personalized sales emails using LLM through Databricks Model Serving.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Databricks CLI configured with authentication (`databricks auth login`)
- Environment variables in `.env` file (see `env.template`)

## Quick Start

```bash
# Install dependencies
make install

# Start both backend (port 8000) and frontend (port 5173)
make start

# Deploy to Databricks apps
make deploy
```

## Project Structure

```
/
├── databricks-app/     # FastAPI backend
│   ├── app.py         # Main API endpoints
│   └── llm_utils.py   # LLM integration logic
├── frontend/          # React (Vite) frontend
├── deploy.sh         # Deployment script
└── Makefile          # Build commands
```

## Usage

1. Open http://localhost:5173 in your browser
2. Select a company or paste customer JSON data
3. Click "Generate Email" to create personalized email content
4. Provide feedback with thumbs up/down

## Deployment

```bash
make deploy
```

## API Endpoints

- `POST /api/generate-email` - Generate email from customer data
- `POST /api/generate-email-stream` - Stream email generation
- `GET /api/companies` - List available companies
- `POST /api/feedback` - Submit user feedback 
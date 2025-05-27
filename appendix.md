# Appendix: Detailed Setup and Prerequisites

## Prerequisites

*   Python 3.8+
*   Node.js 18+ and npm (or yarn)
*   `uv` (for Python package management and virtual environment)
*   Databricks CLI configured with credentials that can access the specified model (`agents-demo-gpt4o`) OR `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables set.
    *   If Databricks SDK cannot be initialized, the backend will use a mock LLM response.

## Setup

### 1. Backend Setup

Navigate to the `backend` directory:
```bash
cd backend
```

Create a virtual environment (recommended):
```bash
uv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install Python dependencies:
```bash
uv sync # Installs from pyproject.toml
```

### 2. Frontend Setup

Navigate to the `frontend` directory (from the project root):
```bash
cd frontend
```

Install Node.js dependencies:
```bash
npm install
# or if you use yarn:
# yarn install
``` 
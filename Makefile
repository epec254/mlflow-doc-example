# Load environment variables from .env file
include .env
export

start:
	@echo "Starting backend server..."
	@(source .venv/bin/activate && cd databricks-app && uvicorn app:app --reload --port 8000) &
	@echo "Starting frontend development server..."
	@(cd frontend && npm run dev)

deploy:
	@echo "Running deployment script..."
	@bash deploy.sh

.PHONY: start install backend_install frontend_install deploy

install: backend_install frontend_install
	@echo "Project setup complete. Backend and frontend dependencies installed."

backend_install:
	@echo "--- Setting up backend ---"
	@echo "Creating virtual environment with uv..." && \
	  uv venv && \
	  echo "Installing Python dependencies with uv from backend/pyproject.toml..." && \
	  uv pip sync pyproject.toml
	@echo "--- Backend setup complete ---"

frontend_install:
	@echo "--- Setting up frontend ---"
	@cd frontend && \
	  echo "Installing Node.js dependencies with npm..." && \
	  npm install
	@echo "--- Frontend setup complete ---" 
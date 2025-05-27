start:
	@echo "Starting backend server..."
	@(cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000) &
	@echo "Starting frontend development server..."
	@(cd frontend && npm run dev)

.PHONY: start install backend_install frontend_install

install: backend_install frontend_install
	@echo "Project setup complete. Backend and frontend dependencies installed."

backend_install:
	@echo "--- Setting up backend ---"
	@cd backend && \
	  echo "Creating virtual environment with uv..." && \
	  uv venv && \
	  echo "Installing Python dependencies with uv from pyproject.toml..." && \
	  uv pip sync pyproject.toml
	@echo "--- Backend setup complete ---"

frontend_install:
	@echo "--- Setting up frontend ---"
	@cd frontend && \
	  echo "Installing Node.js dependencies with npm..." && \
	  npm install
	@echo "--- Frontend setup complete ---" 
# Personalized Email Generation App

This application consists of a FastAPI backend and a React (Vite) frontend to generate personalized emails based on customer data using an LLM.

## Project Structure

```
/
├── backend/
│   ├── main.py         # FastAPI application
│   └── pyproject.toml  # Python dependencies
├── frontend/
│   ├── public/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── input_data.jsonl      # Example input data (not directly used by the app after initial setup)
├── notebook.py           # Original Jupyter notebook
└── README.md             # This file
```

## Prerequisites

Refer to [appendix.md](appendix.md) for detailed prerequisites.

*   A `Makefile` in the root directory with the following targets:
    *   `install`: Installs backend and frontend dependencies.
    *   `run_backend`: Starts the backend server.
    *   `run_frontend`: Starts the frontend development server.
    *   `start`: Starts both backend and frontend servers.

## Setup & Running the Application with Make

1.  **Install Dependencies:**
    Open your terminal in the project root and run:
    ```bash
    make install
    ```
    This command will set up the necessary environments and install dependencies for both the backend and frontend. For detailed manual setup steps, see [appendix.md](appendix.md).

2.  **Run the Application:**
    To start both the backend and frontend servers, run:
    ```bash
    make start
    ```
    Alternatively, you can run them separately:

    *   **Start the Backend Server:**
        In a terminal, from the project root:
        ```bash
        make run_backend
        ```
        The backend server will start on `http://localhost:8000`.

    *   **Start the Frontend Development Server:**
        In a new terminal, from the project root:
        ```bash
        make run_frontend
        ```
        The React development server will start, typically on `http://localhost:5173`.

For more detailed setup instructions, please refer to the [appendix.md](appendix.md) file.

## Running the Application

### 1. Start the Backend Server

In the `backend` directory (with the virtual environment activated):
```bash
uvicorn main:app --reload --port 8000
```
The backend server will start on `http://localhost:8000`.
It will print a message indicating whether the Databricks SDK and OpenAI client were initialized successfully or if it's using the mock client.

### 2. Start the Frontend Development Server

In a new terminal, navigate to the `frontend` directory:
```bash
npm run dev
# or if you use yarn:
# yarn dev
```
The React development server will start, typically on `http://localhost:5173` (this is configured in `vite.config.js`). Your browser should open automatically, or you can navigate to the URL shown in your terminal.

## How to Use

1.  Once both servers are running, open the frontend URL in your browser (usually `http://localhost:5173`).
2.  The application will show a textarea pre-filled with example customer JSON data.
3.  You can modify this data or paste new customer data in the same JSON format.
4.  Click the "Generate Email" button.
5.  The generated email subject and body will appear below the input area.
6.  Check the backend server logs for information about the LLM calls (real or mock).

## Notes

*   **CORS**: The FastAPI backend is configured to allow requests from `http://localhost:5173` (the default Vite dev server port). If your frontend runs on a different port, update `allow_origins` in `backend/main.py` and the `server.port` in `frontend/vite.config.js`.
*   **Databricks SDK**: The backend `main.py` attempts to initialize the Databricks SDK and an OpenAI client using `WorkspaceClient()`. This relies on your environment being set up correctly for Databricks authentication (e.g., `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, or a configured Databricks CLI profile). If it fails, a mock client is used, which returns a pre-defined email. The frontend will display a status message indicating whether the real or mock client is active.
*   **Model Name**: The LLM model is hardcoded as `agents-demo-gpt4o` in `backend/main.py`. Change this if you use a different model endpoint.
*   **Error Handling**: Basic error handling is in place. The frontend will display errors returned from the backend or if the JSON input is invalid. Check browser console and backend terminal for detailed error messages. 
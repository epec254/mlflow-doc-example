from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

# Import from the new llm_utils module
from llm_utils import core_generate_email_logic, PROMPT_V2, openai_client

# The following has been moved to backend/llm_utils.py:
# - Databricks SDK and OpenAI client initialization
# - PROMPT_V2 definition
# - core_generate_email_logic function definition


class EmailRequest(BaseModel):
    customer_info: dict


class EmailOutput(BaseModel):
    subject_line: str
    body: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],  # Adjust if your React app runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate-email/", response_model=EmailOutput)
async def api_generate_email(request_data: EmailRequest):
    customer_data_dict = request_data.customer_info
    try:
        # Use the imported function and prompt
        email_json = core_generate_email_logic(customer_data_dict, PROMPT_V2)
        if (
            not isinstance(email_json, dict)
            or "subject_line" not in email_json
            or "body" not in email_json
        ):
            malformed_detail = f"LLM output parsed but is not in the expected format (missing 'subject_line' or 'body'). Received: {email_json}"
            print(malformed_detail)
            raise HTTPException(status_code=500, detail=malformed_detail)
        return EmailOutput(**email_json)
    except HTTPException:
        raise  # Re-throw HTTPException from core_generate_email_logic or this handler
    except Exception as e:
        print(f"Unexpected error in api_generate_email: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        # Use the imported openai_client for the health check
        "openai_client_initialized": openai_client is not None,
    }


@app.get("/api/env-check")
async def env_check():
    """Endpoint to verify environment variables are loaded correctly"""
    env_vars = {
        "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_EXPERIMENT_ID": os.getenv("MLFLOW_EXPERIMENT_ID"),
        # Don't expose the actual token, just check if it exists
        "DATABRICKS_TOKEN": "***" if os.getenv("DATABRICKS_TOKEN") else None,
    }
    return {
        "status": "ok",
        "environment_variables": env_vars,
        "all_vars_present": all(v is not None for v in env_vars.values()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

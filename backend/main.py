from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

# Import from the new llm_utils module
from llm_utils import core_generate_email_logic, PROMPT_V2, openai_client

# Get model name from environment variable with a default fallback
LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_MODEL:
    raise ValueError("LLM_MODEL environment variable is not set")


# Load customer data from input_data.jsonl
def load_customer_data():
    customers = []
    try:
        with open("../input_data.jsonl", "r") as f:
            for line in f:
                customers.append(json.loads(line))
    except FileNotFoundError:
        # Try alternative path if backend is run from different directory
        try:
            with open("input_data.jsonl", "r") as f:
                for line in f:
                    customers.append(json.loads(line))
        except FileNotFoundError:
            print("Warning: input_data.jsonl not found")
    return customers


CUSTOMER_DATA = load_customer_data()

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
        email_json = core_generate_email_logic(
            customer_data_dict, PROMPT_V2, model=LLM_MODEL
        )
        if (
            not isinstance(email_json, dict)
            or "subject_line" not in email_json
            or "body" not in email_json
        ):
            raise ValueError(
                "LLM output is not in the expected format (missing 'subject_line' or 'body')"
            )
        return EmailOutput(**email_json)
    except Exception as e:
        error_msg = str(e)
        if "OpenAI client not available" in error_msg:
            status_code = 503
        elif "Failed to parse LLM output" in error_msg:
            status_code = 500
        else:
            status_code = 500
        raise HTTPException(status_code=status_code, detail=error_msg)


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
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        # Don't expose the actual token, just check if it exists
        "DATABRICKS_TOKEN": "***" if os.getenv("DATABRICKS_TOKEN") else None,
    }
    return {
        "status": "ok",
        "environment_variables": env_vars,
        "all_vars_present": all(v is not None for v in env_vars.values()),
    }


@app.get("/api/companies")
async def get_companies():
    """Get list of all company names"""
    companies = [{"name": customer["account"]["name"]} for customer in CUSTOMER_DATA]
    return sorted(companies, key=lambda x: x["name"])


@app.get("/api/customer/{company_name}")
async def get_customer_by_name(company_name: str):
    """Get customer data by company name"""
    for customer in CUSTOMER_DATA:
        if customer["account"]["name"] == company_name:
            return customer
    raise HTTPException(status_code=404, detail=f"Company '{company_name}' not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

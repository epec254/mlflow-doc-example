from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

# Attempt to import databricks.sdk and initialize WorkspaceClient
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import DatabricksError
except ImportError:
    print(
        "databricks-sdk not found. Please install it: pip install databricks-sdk[openai]"
    )
    WorkspaceClient = None
    DatabricksError = None
    openai_client = None

# Initialize OpenAI client (once)
# This assumes Databricks environment is configured (e.g., DATABRICKS_HOST, DATABRICKS_TOKEN)
# or databricks-cli is configured.
if WorkspaceClient:
    try:
        w = WorkspaceClient()  # Auto-configures from environment or ~/.databrickscfg
        openai_client = w.serving_endpoints.get_open_ai_client()
        print("Successfully initialized Databricks SDK and OpenAI client.")
    except DatabricksError as e:
        print(f"Error initializing Databricks SDK: {e}")
        print(
            "Please ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set or databricks-cli is configured."
        )
        openai_client = None
    except Exception as e:
        print(f"An unexpected error occurred during Databricks SDK initialization: {e}")
        openai_client = None
else:
    openai_client = None

PROMPT_V2 = """
You are an expert sales communication assistant for CloudFlow Inc. Your task is to generate a personalized, professional follow-up email for our sales representatives to send to their customers at the end of the day.

## INPUT DATA
You will be provided with a JSON object containing:
- Account information
- Recent activity data (meetings, product usage, support tickets)
- Sales representative details

## EMAIL REQUIREMENTS
Generate an email that follows these guidelines:

1. SUBJECT LINE:
   - Concise and specific to the most important update or follow-up point
   - Include the company name if appropriate

2. GREETING:
   - Address the main contact by first name
   - Use a professional but friendly opening

3. BODY CONTENT (prioritize in this order):
   - Reference the most recent meeting/interaction and acknowledge key points discussed
   - Discuss any pressing issues that are still open immediatly afterwards
   - Provide updates on any urgent or recently resolved support tickets
   - Highlight positive product usage trends or achievements
   - Address any specific action items from previous meetings
   - Include personalized recommendations based on features listed as 'least_used_features' and directly related to the 'potential_opportunity' field.
      - Make sure these recommendations can NOT be copied to another customer in a different situation
      - No more than ONE feature recommendation for accounts with open critical issues
   - Suggest clear and specific next steps
      - Only request a meeting if it can be tied to specific action items


4. TONE AND STYLE:
   - Professional but conversational
   - Concise paragraphs (2-3 sentences each)
   - Use bullet points for lists or multiple items
   - Balance between being informative and actionable
   - Personalized to reflect the existing relationship
   - Adjust formality based on the customer's industry and relationship history

5. CLOSING:
   - Include an appropriate sign-off
   - Use the sales rep's signature from the provided data
   - No generic marketing language or overly sales-focused calls to action

## OUTPUT FORMAT
Provide the complete email as JUST a JSON object that can be loaded via `json.loads()` (do not wrap the JSON in backticks) with:
- `subject_line`: Subject line
- `body`: Body content with appropriate spacing and formatting including the signature

Remember, this email should feel like it was thoughtfully written by the sales representative based on their specific knowledge of the customer, not like an automated message."""


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


def core_generate_email_logic(customer_data: dict, prompt_template: str):
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not available. Please check backend server logs for Databricks SDK initialization issues.",
        )

    try:
        response = openai_client.chat.completions.create(
            model="agents-demo-gpt4o",  # Model from notebook
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": json.dumps(customer_data)},
            ],
        )
        s = response.choices[0].message.content
    except Exception as e:
        # Catch issues with the actual API call
        print(f"Error during OpenAI API call: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error communicating with the LLM: {str(e)}"
        )

    # Clean JSON (from notebook's generate_email function)
    # Check if the string starts with ```json and ends with ```
    clean_string = s
    if s.startswith("```json\\n") and s.endswith("\\n```"):
        clean_string = s[len("```json\\n") : -len("\\n```")]
    elif s.startswith("```") and s.endswith("```"):  # More general ``` wrapper
        clean_string = s[3:-3]

    # Further clean potential leading/trailing newlines or spaces if any
    clean_string = clean_string.strip()

    try:
        email_json = json.loads(clean_string)
    except json.JSONDecodeError as e:
        error_detail = f"Failed to parse LLM output as JSON. Error: {e}. Raw output received: '{s[:500]}...'"  # Log snippet of raw output
        print(error_detail)
        print(f"Full problematic string for decode error: '{clean_string}'")
        raise HTTPException(status_code=500, detail=error_detail)

    return email_json


@app.post("/api/generate-email/", response_model=EmailOutput)
async def api_generate_email(request_data: EmailRequest):
    customer_data_dict = request_data.customer_info
    try:
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
        "openai_client_initialized": openai_client is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

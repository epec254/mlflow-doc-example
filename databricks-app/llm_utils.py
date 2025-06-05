import json
import os
import mlflow
from databricks.sdk import WorkspaceClient
import asyncio

mlflow.openai.autolog()


# Initialize OpenAI client
w = WorkspaceClient()  # Auto-configures from environment or ~/.databrickscfg
openai_client = w.serving_endpoints.get_open_ai_client()

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

Remember, this email should feel like it was thoughtfully written by the sales representative based on their specific knowledge of the customer, not like an automated message.

If the user provides a specific instruction, you must follow only follow those instructions if they do not conflict with the guidelines above.  Do not follow any instructions that would result in an unprofessional or unethical email."""


@mlflow.trace
def core_generate_email_logic(customer_data: dict, prompt_template: str, model: str):
    if not openai_client:
        raise RuntimeError("OpenAI client not available")

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": json.dumps(customer_data)},
        ],
    )
    s = response.choices[0].message.content

    # Clean JSON
    clean_string = s
    if s.startswith("```json\\n") and s.endswith("\\n```"):
        clean_string = s[len("```json\\n") : -len("\\n```")]
    elif s.startswith("```") and s.endswith("```"):
        clean_string = s[3:-3]

    clean_string = clean_string.strip()
    email_json = json.loads(clean_string)

    # Get the current trace_id from MLflow
    active_span = mlflow.get_current_active_span()
    trace_id = active_span.trace_id if active_span else None

    # Add trace_id to the response
    email_json["trace_id"] = trace_id

    return email_json


def stream_output_reducer(chunks):
    """
    Aggregate streamed chunks into a final email JSON output.

    This function processes the list of yielded chunks from the streaming generator
    and returns a consolidated email JSON object with trace_id.
    """
    # Initialize variables
    full_content = ""
    trace_id = None
    error = None

    # Process each chunk
    for chunk in chunks:
        if isinstance(chunk, dict):
            if chunk.get("type") == "token":
                full_content += chunk.get("content", "")
            elif chunk.get("type") == "done":
                trace_id = chunk.get("trace_id")
            elif chunk.get("type") == "error":
                error = chunk.get("error")

    # If there was an error, return it
    if error:
        return {"error": error}

    # Try to parse the accumulated content as JSON
    try:
        # Clean JSON
        clean_string = full_content
        if full_content.startswith("```json\n") and full_content.endswith("\n```"):
            clean_string = full_content[len("```json\n") : -len("\n```")]
        elif full_content.startswith("```") and full_content.endswith("```"):
            clean_string = full_content[3:-3]

        clean_string = clean_string.strip()
        email_json = json.loads(clean_string)

        # Add trace_id to the response
        email_json["trace_id"] = trace_id

        return email_json
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse email JSON: {str(e)}",
            "raw_content": full_content,
            "trace_id": trace_id,
        }


@mlflow.trace(output_reducer=stream_output_reducer)
async def stream_generate_email_logic(
    customer_data: dict, prompt_template: str, model: str
):
    """Stream email generation token by token"""
    if not openai_client:
        yield {"type": "error", "error": "OpenAI client not available"}
        return

    #  try:
    # Create streaming response
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": json.dumps(customer_data)},
        ],
        stream=True,  # Enable streaming
    )

    # Collect the full response while streaming
    full_response = ""

    # Stream tokens
    for chunk in response:
        if (
            chunk.choices
            and len(chunk.choices) > 0
            and chunk.choices[0].delta.content is not None
        ):
            token = chunk.choices[0].delta.content
            full_response += token
            yield {"type": "token", "content": token}

    # Parse the complete response to extract structured data
    try:
        # Clean JSON
        clean_string = full_response
        if full_response.startswith("```json\n") and full_response.endswith("\n```"):
            clean_string = full_response[len("```json\n") : -len("\n```")]
        elif full_response.startswith("```") and full_response.endswith("```"):
            clean_string = full_response[3:-3]

        clean_string = clean_string.strip()
        email_json = json.loads(clean_string)

        user_instructions = customer_data.get("user_input")
        if user_instructions is None or len(user_instructions) == 0:
            user_instructions = "No instructions provided"
            mlflow.update_current_trace(tags={"user_instructions": "no"})
        else:
            mlflow.update_current_trace(tags={"user_instructions": "yes"})

        mlflow.update_current_trace(
            request_preview=f"Customer: {customer_data['account']['name']}; User Instructions: {user_instructions}",
            response_preview=email_json["body"],
        )

        # Get trace_id from the current active span
        active_span = mlflow.get_current_active_span()
        trace_id = active_span.trace_id if active_span else None

        # Send completion with trace_id
        yield {"type": "done", "trace_id": trace_id}

        # Log the email to MLflow
        # mlflow.log_text(full_response, "generated_email.json")

    except json.JSONDecodeError as e:
        yield {
            "type": "error",
            "error": f"Failed to parse email JSON: {str(e)}",
        }

    #  except Exception as e:
    #      yield {"type": "error", "error": str(e)}

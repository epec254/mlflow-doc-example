import json
import mlflow
from dotenv import load_dotenv
import os
import functools
from mlflow.genai import scorers
from mlflow.genai.scorers import scorer
from databricks.agents.evals import judges
from mlflow.genai.evaluation.base import _evaluate, _to_predict_fn
from mlflow.entities import Feedback
from databricks.sdk import WorkspaceClient
import llm_utils

# Load environment variables from .env file
load_dotenv()


def load_input_data(file_path: str) -> list:
    """
    Load input data from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing customer data

    Returns:
        List of dictionaries containing customer information in the format:
        [{"inputs": {"customer_info": {...}}}, ...]
    """
    with open(file_path, "r") as file:
        return [{"inputs": {"customer_info": json.loads(line)}} for line in file]


# Initialize workspace client
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

# Define evaluation guidelines
guidelines = {
    "accuracy": """The response correctly references all factual information from the provided_info based on these rules:
- All factual information must be directly sourced from the provided data with NO fabrication
- Names, dates, numbers, and company details must be 100% accurate with no errors
- Meeting discussions must be summarized with the exact same sentiment and priority as presented in the data
- Support ticket information must include correct ticket IDs, status, and resolution details when available
- All product usage statistics must be presented with the same metrics provided in the data
- No references to CloudFlow features, services, or offerings unless specifically mentioned in the customer data
- AUTOMATIC FAIL if any information is mentioned that is not explicitly provided in the data""",
    "personalized": """The response demonstrates clear personalization based on the provided_info based on these rules:
- Email must begin by referencing the most recent meeting/interaction
- Immediatly next, the email must address the customer's MOST pressing concern as evidenced in the data
- Content structure must be customized based on the account's health status (critical issues first for "Fair" or "Poor" accounts)
- Industry-specific language must be used that reflects the customer's sector
- Recommendations must ONLY reference features that are:
  a) Listed as "least_used_features" in the data, AND
  b) Directly related to the "potential_opportunity" field
- Relationship history must be acknowledged (new vs. mature relationship)
- Deal stage must influence communication approach (implementation vs. renewal vs. growth)
- AUTOMATIC FAIL if recommendations could be copied to another customer in a different situation""",
    "relevance": """The response prioritizes content that matters to the recipient in the provided_info based on these rules:
- Critical support tickets (status="Open (Critical)") must be addressed after the greeting, reference to the most recent interaction, any pleasantrys, and references to closed tickets
    - it is ok if they name is slightly different as long as it is clearly the same issue as in the provided_info
- Time-sensitive action items must be addressed before general updates
- Content must be ordered by descending urgency as defined by:
  1. Critical support issues
  2. Action items explicitly stated in most recent meeting
  3. Upcoming renewal if within 30 days
  4. Recently resolved issues
  5. Usage trends and recommendations
- No more than ONE feature recommendation for accounts with open critical issues
- No mentions of company news, product releases, or success stories not directly requested by the customer
- No calls to action unrelated to the immediate needs in the data
- AUTOMATIC FAIL if the email requests a meeting without being tied to a specific action item or opportunity in the data""",
    "follows_instructions": """The response must strictly adhere to any specific user-provided instructions, even when they conflict with other prompt guidelines based on these rules:
- User instructions take absolute precedence over all other prompt requirements
- If user instructions specify a particular tone, format, or content structure, follow it exactly regardless of other guidelines
- If user instructions request specific information to be included or excluded, honor those requests completely
- If user instructions conflict with standard email best practices or other evaluation criteria, prioritize the user instructions
- If user instructions specify a particular greeting, closing, or signature format, use exactly what was requested
- If user instructions provide specific language or phrases to include, incorporate them verbatim
- If user instructions request deviation from the standard email structure or content priorities, follow the user's preferred approach
- AUTOMATIC FAIL if any explicit user instruction is ignored or overridden by other prompt guidelines
- NOTE: This only applies when specific user instructions are provided - general prompt guidelines still apply when no conflicting user instructions exist""",
}


@scorer
def grounded(inputs, outputs):
    """Evaluate if the response is grounded in the provided information."""
    assessment = judges.groundedness(
        request="Write an email for this customer.",
        response=outputs["body"],
        retrieved_context=[{"content": json.dumps(inputs["customer_info"])}],
    )
    return Feedback(
        name="grounded", value=assessment.value, rationale=assessment.rationale
    )


@scorer
def email_guidelines(inputs, outputs):
    """Evaluate if the email follows the defined guidelines."""
    results = []
    for guideline_name, guideline in guidelines.items():
        output = judges.guideline_adherence(
            request="Write an email for this customer.",
            guidelines=[guideline],
            response=outputs["body"],
            guidelines_context={"provided_info": json.dumps(inputs["customer_info"])},
        )
        results.append(
            Feedback(
                name=guideline_name, value=output.value, rationale=output.rationale
            )
        )
    return results


@scorer
def rep_name_in_email(inputs, outputs):
    """Check if the sales representative's name is included in the email."""
    return inputs["customer_info"]["sales_rep"]["name"] in outputs["body"]


def predict_fn(customer_info, prompt_template, model):
    """
    Prediction function that uses core_generate_email_logic to generate emails.

    Args:
        input_data: Dictionary containing customer information
        prompt_template: The prompt template to use for generation
        model: The model to use for generation

    Returns:
        Dictionary containing generated email subject and body
    """
    # Temporarily override the constants in llm_utils for evaluation
    original_prompt = llm_utils.PROMPT
    original_model = llm_utils.LLM_MODEL

    try:
        llm_utils.PROMPT = prompt_template
        llm_utils.LLM_MODEL = model

        return llm_utils.core_generate_email_logic(customer_data=customer_info)
    finally:
        # Restore original values
        llm_utils.PROMPT = original_prompt
        llm_utils.LLM_MODEL = original_model


def evaluate_email_generation(prompt, model):
    """
    Evaluate email generation using MLflow's evaluation harness.

    Args:
        data: List of input data for evaluation

    Returns:
        Evaluation results from MLflow
    """
    # Create a partial function with prompt_template and model pre-filled
    partial_predict_fn = functools.partial(
        predict_fn, prompt_template=prompt, model=model
    )

    # Load evaluation data
    data = load_input_data("../input_data.jsonl")

    # Run evaluation
    return _evaluate(
        data=data[:5],
        predict_fn=partial_predict_fn,
        scorers=[scorers.safety, email_guidelines, grounded, rep_name_in_email],
    )


def eval_v1():

    prompt = """You are an expert sales communication assistant for CloudFlow Inc. Your task is to generate a personalized, professional follow-up email for our sales representatives to send to their customers at the end of the day.

    ## INPUT DATA
    You will be provided with a JSON object containing:
    - Account information
    - Recent activity data (meetings, product usage, support tickets)
    - Sales representative details

    ## EMAIL REQUIREMENTS
    Generate an email that follows these guidelines:

    1. SUBJECT LINE:
    - Engaging and attention-grabbing
    - Include the company name if appropriate

    2. GREETING:
    - Address the main contact by first name
    - Use a professional but friendly opening

    3. BODY CONTENT:
    - Begin with a brief mention of CloudFlow's recent improvements
    - Reference the most recent meeting/interaction
    - Provide updates on support tickets
    - Highlight positive product usage trends
    - Address any action items from previous meetings
    - Include recommendations for additional features they should try
    - Suggest scheduling a follow-up meeting soon

    4. TONE AND STYLE:
    - Professional but enthusiastic
    - Show expertise by using industry terminology
    - Include at least one customer success story
    - Balance being informative with driving future business
    - Personalized where possible
    - Ensure email is comprehensive enough to cover all important points

    5. CLOSING:
    - Include an appropriate sign-off
    - Use the sales rep's signature from the provided data
    - Add a brief mention of an upcoming product release or feature

    ## OUTPUT FORMAT
    Provide the complete email as JUST a JSON object that can be loaded via `json.loads()` (do not wrap the JSON in backticks) with:
    - `subject_line`: Subject line
    - `body`: Body content with appropriate spacing and formatting including the signature

    Remember, this email should position the sales representative as a trusted advisor who can help the customer get maximum value from CloudFlow's solutions."""

    evaluate_email_generation(prompt=prompt, model="databricks-claude-3-7-sonnet")


def eval_v2():
    prompt = """
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

    evaluate_email_generation(prompt=prompt, model="databricks-claude-3-7-sonnet")


if __name__ == "__main__":
    eval_v1()
    eval_v2()

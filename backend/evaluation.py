import json
import mlflow
from dotenv import load_dotenv
import os
from mlflow.genai import scorers
from mlflow.genai.scorers import scorer
from databricks.agents.evals import judges
from mlflow.genai.evaluation.base import _evaluate, _to_predict_fn
from mlflow.entities import Feedback
from databricks.sdk import WorkspaceClient
from llm_utils import core_generate_email_logic, PROMPT_V2

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


def predict_fn(customer_info):
    """
    Prediction function that uses core_generate_email_logic to generate emails.

    Args:
        input_data: Dictionary containing customer information

    Returns:
        Dictionary containing generated email subject and body
    """
    return core_generate_email_logic(
        customer_data=customer_info,
        prompt_template=prompt,
        model="databricks-claude-3-7-sonnet",
    )


def evaluate_email_generation(data):
    """
    Evaluate email generation using MLflow's evaluation harness.

    Args:
        data: List of input data for evaluation

    Returns:
        Evaluation results from MLflow
    """
    return _evaluate(
        data=data[:5],
        predict_fn=predict_fn,
        scorers=[scorers.safety, email_guidelines, grounded, rep_name_in_email],
    )


def main():
    data = load_input_data("../input_data.jsonl")
    results = evaluate_email_generation(data)


if __name__ == "__main__":
    main()

import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


# This function should be the same as what is called by your production application.
# It will be called by `evaluate(...)`.
@mlflow.trace
def my_app(customer_name: str, last_conversation_details: str, known_issues: str):
    system_prompt = """Please generate an email for a sales rep to send based on the provided data"""

    user_message = f"""Customer name: {customer_name}\nDetails of the last converastion: {last_conversation_details}\nKnown support issues: {known_issues}"""

    response = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return {"email": response.choices[0].message.content}


# Test the predict_fn
output = my_app(
    **{
        "customer_name": "Acme Corporation",
        "last_conversation_details": "Met with CTO Sarah Johnson who expressed interest in our enterprise security features. She mentioned budget approval in Q3.",
        "known_issues": "Their IT team reported slow response times with our API during peak hours.",
    }
)
print(output)

# Evaluation dataset
eval_data = [
    {
        "inputs": {
            "customer_name": "Acme Corporation",
            "last_conversation_details": "Met with CTO Sarah Johnson who expressed interest in our enterprise security features. She mentioned budget approval in Q3.",
            "known_issues": "Their IT team reported slow response times with our API during peak hours.",
        }
    },
    {
        "inputs": {
            "customer_name": "TechNova Solutions",
            "last_conversation_details": "Product demo with their engineering team last Thursday. They were particularly impressed with the scalability options.",
            "known_issues": "Integration with their legacy systems has been challenging. Support ticket #4532 is still open.",
        }
    },
    {
        "inputs": {
            "customer_name": "Global Retail Partners",
            "last_conversation_details": "Conference call with procurement director Marcus Lee who requested pricing for 500+ user licenses.",
            "known_issues": "Mobile app crashes reported by several users in their organization after our latest update.",
        }
    },
    {
        "inputs": {
            "customer_name": "Healthcare Innovations",
            "last_conversation_details": "On-site meeting with their compliance team regarding HIPAA requirements and our data protection measures.",
            "known_issues": "Concerns about data migration timeline and potential service interruptions.",
        }
    },
    {
        "inputs": {
            "customer_name": "EduTech Systems",
            "last_conversation_details": "Virtual presentation to their board about our new learning management features. Decision maker Dr. Williams asked for case studies.",
            "known_issues": "SSO implementation has been problematic for their IT department.",
        }
    },
]

# Define scorers

# 1. Use a predefined scorer to evaluate if the LLM is provides a relevant response
from mlflow.genai.scorers import relevance_to_query

# 2. Create a guideline-based LLM judge to evaluate against our use case's specific criteria for how the email should be written
from mlflow.genai.scorers import guideline_adherence

email_style = guideline_adherence.with_config(
    name="email_style",
    global_guidelines=[
        "The email must be polite and professional.  If there are any open issues, they should be referenced first after any pleasantries."
    ],
)

# 3. Create a custom scorer determinstically evaluate if the LLM correctly mentions the company name in the generated email
from mlflow.genai.scorers import scorer


@scorer
def company_name_in_email(inputs, outputs):
    """Check if the company name is included in the email."""
    return inputs["customer_name"] in outputs["email"]


# Run evaluation
from mlflow.genai import evaluate


mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_app,
    scorers=[company_name_in_email, email_style, relevance_to_query],
)

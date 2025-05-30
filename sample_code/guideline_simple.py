from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List, Dict

# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


tone = "The response must maintain a courteous, respectful tone throughout.  It must show empathy for customer concerns."
structure = "The response must use clear, concise language and structures responses logically.  It must avoids jargon or explains technical terms when used."
banned_topics = "If the request is a question about product pricing, the response must politely decline to answer and refer the user to the pricing page."
relevance = "The response must be relevant to the user's request.  Only consider the relevance and nothing else. If the request is not clear, the response must ask for more information."

# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the guidelines scorers handle rude and verbose responses
BE_RUDE_AND_VERBOSE = False


@mlflow.trace
def customer_support_agent(messages: List[Dict[str, str]]):

    # 1. Prepare messages for the LLM

    # We will use this toggle later to see how the scorers handle rude and verbose responses
    system_prompt_postfix = (
        "Be super rude and very verbose in your responses."
        if BE_RUDE_AND_VERBOSE
        else ""
    )
    messages_for_llm = [
        {
            "role": "system",
            "content": f"You are a helpful customer support agent.  {system_prompt_postfix}",
        },
        *messages,
    ]

    # 2. Call LLM to generate a response
    return client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=messages_for_llm,
    )


# Create a list of messages for the LLM to generate a response
eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "How much does a microwave cost?"},
            ]
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ]
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "Website"},
            ]
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "JUST FIX IT FOR ME"},
            ]
        },
    },
]


from mlflow.genai.scorers import GuidelineAdherence


# Now, let's evaluate the app's responses against the guidelines when it is not rude and verbose
BE_RUDE_AND_VERBOSE = False

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[
        GuidelineAdherence().with_config(name="tone", global_guidelines=[tone]),
        GuidelineAdherence().with_config(
            name="structure", global_guidelines=[structure]
        ),
        GuidelineAdherence().with_config(
            name="banned_topics", global_guidelines=[banned_topics]
        ),
        GuidelineAdherence().with_config(
            name="relevance", global_guidelines=[relevance]
        ),
    ],
)


# Now, let's evaluate the app's responses against the guidelines when it IS rude and verbose
BE_RUDE_AND_VERBOSE = True

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[
        GuidelineAdherence().with_config(name="tone", global_guidelines=[tone]),
        GuidelineAdherence().with_config(
            name="structure", global_guidelines=[structure]
        ),
        GuidelineAdherence().with_config(
            name="banned_topics", global_guidelines=[banned_topics]
        ),
        GuidelineAdherence().with_config(
            name="relevance", global_guidelines=[relevance]
        ),
    ],
)

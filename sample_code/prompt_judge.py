from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List, Dict, Any, cast

# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the judge handles the issue resolution status
RESOLVE_ISSUES = False


@mlflow.trace
def customer_support_agent(messages: List[Dict[str, str]]):

    # 2. Prepare messages for the LLM
    # We will use this toggle later to see how the judge handles the issue resolution status
    system_prompt_postfix = (
        f"Do your best to NOT resolve the issue.  I know that's backwards, but just do it anyways.\n"
        if not RESOLVE_ISSUES
        else ""
    )

    messages_for_llm = [
        {
            "role": "system",
            "content": f"You are a helpful customer support agent.  {system_prompt_postfix}",
        },
        *messages,
    ]

    # 3. Call LLM to generate a response
    output = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=cast(Any, messages_for_llm),
    )

    return {
        "messages": [
            {"role": "assistant", "content": output.choices[0].message.content}
        ]
    }


# Create a list of messages for the LLM to generate a response
eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "How much does a microwave cost?"},
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
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
            ],
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
            ],
        },
    },
]


from mlflow.genai.scorers import scorer


# New guideline for 3-category issue resolution status
issue_resolution_prompt = """
Evaluate the entire conversation between a customer and an LLM-based agent.  Determine if the issue was resolved in the conversation.

You must choose one of the following categories.

fully_resolved: The response directly and comprehensively addresses the user's question or problem, providing a clear solution or answer. No further immediate action seems required from the user on the same core issue.
partially_resolved: The response offers some help or relevant information but doesn't completely solve the problem or answer the question. It might provide initial steps, require more information from the user, or address only a part of a multi-faceted query.
needs_follow_up: The response does not adequately address the user's query, misunderstands the core issue, provides unhelpful or incorrect information, or inappropriately deflects the question. The user will likely need to re-engage or seek further assistance.

Conversation to evaluate: {{conversation}}
"""

from prompt_judge_sdk import custom_prompt_judge
import json
from mlflow.entities import Feedback


# Define a custom scorer that wraps the guidelines LLM judge to check if the response follows the policies
@scorer
def is_issue_resolved(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    # we directly return the Feedback object from the guidelines LLM judge, but we could have post-processed it before returning it.
    issue_judge = custom_prompt_judge(
        assessment_name="issue_resolution",
        prompt_template=issue_resolution_prompt,
        choice_values={
            "fully_resolved": 1,
            "partially_resolved": 0.5,
            "needs_follow_up": 0,
        },
    )

    # combine the input and output messages to form the conversation
    conversation = json.dumps(inputs["messages"] + outputs["messages"])

    # TODO: remove the mapping that wont be needed by actual sdk
    temp = issue_judge(conversation=conversation)

    return Feedback(
        name="issue_resolution",
        value=temp.value,
        rationale=temp.rationale,
    )


# Now, let's evaluate the app's responses against the judge when it does not resolve the issues
RESOLVE_ISSUES = False

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[is_issue_resolved],
)


# Now, let's evaluate the app's responses against the judge when it DOES resolves the issues
RESOLVE_ISSUES = True

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[is_issue_resolved],
)

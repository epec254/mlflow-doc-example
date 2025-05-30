from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


import os
import mlflow
from openai import OpenAI
from typing import List, Dict, Any
from mlflow.entities.trace import Trace
from mlflow.genai.scorers import scorer


# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


@mlflow.trace
def sample_app(messages: List[Dict[str, str]]):

    # 1. Prepare messages for the LLM
    messages_for_llm = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
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


@scorer
def my_metric(
    inputs: Dict[Any, Any],
    outputs: Dict[Any, Any],
    expectations: Dict[str, Any],
    trace: Trace,
):
    # placeholder return value
    return 1


# eval_results = mlflow.genai.evaluate(
#     data=eval_dataset, predict_fn=sample_app, scorers=[my_metric]
# )

# generated_traces = mlflow.search_traces(
#     experiment_ids=[os.environ.get("MLFLOW_EXPERIMENT_ID")], run_id=eval_results.run_id
# )

generated_traces = mlflow.search_traces(
    experiment_ids=[os.environ.get("MLFLOW_EXPERIMENT_ID")],
    run_id="9391245750fd43e5b20f513b3e8ddbd6",
)


# print(generated_traces)


@scorer
def not_empty(outputs):
    # "yes" for Pass and "no" for Fail.
    return "yes" if outputs["choices"][0]["message"]["content"].strip() != "" else "no"


mlflow.genai.evaluate(data=generated_traces, scorers=[not_empty])

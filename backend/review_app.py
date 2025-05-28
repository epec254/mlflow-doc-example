import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import mlflow
from openai import OpenAI

# import json # Removed json import

from mlflow.entities import Document
from typing import List, Dict  # Dict might not be needed anymore


mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)

## Sample app that we will review outputs from


# Spans of type RETRIEVER are rendered in the Review App as documents.
@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str) -> List[Document]:
    normalized_query = query.lower()
    if "john doe" in normalized_query:
        return [
            Document(
                id="conversation_123",
                page_content="John Doe mentioned issues with login on July 10th. Expressed interest in feature X.",
                metadata={"doc_uri": "http://domain.com/conversations/123"},
            ),
            Document(
                id="conversation_124",
                page_content="Follow-up call with John Doe on July 12th. Login issue resolved. Discussed pricing for feature X.",
                metadata={"doc_uri": "http://domain.com/conversations/124"},
            ),
        ]
    else:
        return [
            Document(
                id="ticket_987",
                page_content="Acme Corp raised a critical P0 bug regarding their main dashboard on July 15th.",
                metadata={"doc_uri": "http://domain.com/tickets/987"},
            )
        ]


@mlflow.trace
def my_app(messages: List[Dict[str, str]]):
    # 1. Retrieve conversations based on the last user message
    last_user_message_content = messages[-1]["content"]
    retrieved_documents = retrieve_docs(query=last_user_message_content)
    retrieved_docs_text = "\n".join([doc.page_content for doc in retrieved_documents])

    # 2. Prepare messages for the LLM
    messages_for_llm = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {
            "role": "user",
            "content": f"Additional retrieved context:\n{retrieved_docs_text}\n\nNow, please provide the one-paragraph summary based on the user's request {last_user_message_content} and this retrieved context.",
        },
    ]

    # 3. Call LLM to generate the summary
    return client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=messages_for_llm,
    )


## Run app to get traces

# use verison tracking to be able to easily query for the traces
tracked_model = mlflow.set_active_model(name="my_app")

sample_messages_1 = [
    {"role": "user", "content": "what issues does john doe have?"},
]
summary1_output = my_app(sample_messages_1)

sample_messages_2 = [
    {"role": "user", "content": "what issues does acme corp have?"},
]
summary2_output = my_app(sample_messages_2)

traces = mlflow.search_traces(model_id=tracked_model.model_id)
print(traces)


from mlflow.genai.labeling import create_labeling_session
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText

# from databricks.sdk.errors import NotFound
# from IPython.display import Markdown

# The review app is tied to the current MLFlow experiment.
# my_app = get_review_app()

# Search for the traces above using the run_id above
# traces = mlflow.search_traces(run_id=run.info.run_id)

summary_quality = create_label_schema(
    name="summary_quality",
    # Type can be "expectation" or "feedback".
    type="feedback",
    title="Is this summary concise and helpful?",
    # see docs for other question formats: https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#label-schemas
    input=InputCategorical(options=["Yes", "No"]),
    instruction="Please provide a rationale below.",
    enable_comment=True,
    overwrite=True,
)

expected_summary = create_label_schema(
    name="expected_summary",
    # Type can be "expectation" or "feedback".
    type="expectation",
    title="Please provide the correct summary for the user's request.",
    # see docs for other question formats: https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#label-schemas
    input=InputText(),
    # instruction="Please provide a rationale below.",
    # enable_comment=True,
    overwrite=True,
)
print(expected_summary)

label_summaries = create_labeling_session(
    name="label_summaries",
    assigned_users=[],
    label_schemas=[summary_quality.name, expected_summary.name],
)

label_summaries.add_traces(traces)

print(f"Share this Review App with your team: {label_summaries.url}")

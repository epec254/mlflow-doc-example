from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List

# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


# Retriever function called by the sample app
@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str) -> List[Document]:
    return [
        Document(
            id="sql_doc_1",
            page_content="SELECT is a fundamental SQL command used to retrieve data from a database. You can specify columns and use a WHERE clause to filter results.",
            metadata={"doc_uri": "http://example.com/sql/select_statement"},
        ),
        Document(
            id="sql_doc_2",
            page_content="JOIN clauses in SQL are used to combine rows from two or more tables, based on a related column between them. Common types include INNER JOIN, LEFT JOIN, and RIGHT JOIN.",
            metadata={"doc_uri": "http://example.com/sql/join_clauses"},
        ),
        Document(
            id="sql_doc_3",
            page_content="Aggregate functions in SQL, such as COUNT(), SUM(), AVG(), MIN(), and MAX(), perform calculations on a set of values and return a single summary value.  The most common aggregate function in SQL is COUNT().",
            metadata={"doc_uri": "http://example.com/sql/aggregate_functions"},
        ),
    ]


# Sample app that we will evaluate
@mlflow.trace
def sample_app(query: str):
    # 1. Retrieve conversations based on the last user message
    retrieved_documents = retrieve_docs(query=query)
    retrieved_docs_text = "\n".join([doc.page_content for doc in retrieved_documents])

    # 2. Prepare messages for the LLM
    messages_for_llm = [
        {
            "role": "system",
            # Fake prompt to show how the various scorers identify quality issues.
            "content": f"Answer the user's question based on the following retrieved context: {retrieved_docs_text}.  Do not mention the fact that provided context exists in your answer.  If the context is not relevant to the question, generate the best response you can.",
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    # 3. Call LLM to generate the summary
    return client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=messages_for_llm,
    )


eval_dataset = [
    {
        "inputs": {"query": "What is the most common aggregate function in SQL?"},
        "expectations": {
            "expected_facts": ["Most common aggregate function in SQL is COUNT()."],
        },
    },
    {
        "inputs": {"query": "How do I use MLflow?"},
        "expectations": {
            "expected_facts": [
                "MLflow is a tool for managing and tracking machine learning experiments."
            ],
        },
    },
]

from mlflow.genai.scorers import (
    Correctness,
    GuidelineAdherence,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
)


# Run predefined scorers that require ground truth
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=sample_app,
    scorers=[
        Correctness(),
        # RelevanceToQuery(),
        # RetrievalGroundedness(),
        # RetrievalRelevance(),
        RetrievalSufficiency(),
        # Safety(),
    ],
)


# Run predefined scorers that do NOT require ground truth
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=sample_app,
    scorers=[
        # Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        # RetrievalSufficiency(),
        Safety(),
    ],
)


# sample_app(query="What is the most common aggregate function in SQL?")
# sample_app(query="How do I use MLflow?")

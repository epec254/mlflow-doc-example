from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List, Dict, Any
from datetime import datetime

# Enable auto logging for OpenAI
mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)

# from mlflow.genai.datasets import create_dataset, get_dataset

# # create_dataset(uc_table_name="agents_demo.email_app.eval_dataset")

# dataset = get_dataset(uc_table_name="agents_demo.email_app.eval_dataset")

# dataset.insert(
#     [
#         {
#             "dataset_record_id": "eric-test",
#             "inputs": {
#                 "messages": [
#                     {"role": "user", "content": "Hello, how are you?2332"},
#                 ]
#             },
#             "last_update_time": "xcc",
#             "source": {"test": "test"},
#         },
#     ]
# )


# # from databrickks.

# # dataset.


from mlflow.genai.judges import is_context_sufficient

# feedback = is_context_sufficient(
#     request="What is the capital of France?",
#     context=[
#         {"content": "Paris is the capital of France."},
#         {"content": "Paris is known for its Eiffel Tower."},
#     ],
#     expected_facts=["Paris is the capital of France."],
# )
# print(feedback.value)  # "yes"


# feedback = is_context_sufficient(
#     request="What is the capital of France?",
#     context="fdgffdgfdf",
#     expected_facts=["Paris is the capital of France."],
# )
# print(feedback.rationale)  # "yes"


from mlflow.genai.judges import is_grounded

# result = is_grounded(
#     request="What is the capital of France?",
#     response="Paris",
#     context="Paris is the capital of France.",
# )

# print(result)

# result = is_grounded(
#     request="What is the capital of France?",
#     response="Paris",
#     context="Paris is known for its Eiffel Tower.",
# )

# print(result)

# Feedback(name='groundedness', source=AssessmentSource(source_type='LLM_JUDGE', source_id='databricks'), trace_id=None, rationale="The response asks 'What is the capital of France?' and answers 'Paris'. The retrieved context states 'Paris is the capital of France.' This directly supports the answer given in the response.", metadata=None, span_id=None, create_time_ms=1748806639995, last_update_time_ms=1748806639995, assessment_id=None, error=None, expectation=None, feedback=FeedbackValue(value=<CategoricalRating.YES: 'yes'>, error=None), overrides=None, valid=True)
# Feedback(name='groundedness', source=AssessmentSource(source_type='LLM_JUDGE', source_id='databricks'), trace_id=None, rationale="The retrieved context states that 'Paris is known for its Eiffel Tower,' but it does not mention that Paris is the capital of France. Therefore, the response is not fully supported by the retrieved context.", metadata=None, span_id=None, create_time_ms=1748806641260, last_update_time_ms=1748806641260, assessment_id=None, error=None, expectation=None, feedback=FeedbackValue(value=<CategoricalRating.NO: 'no'>, error=None), overrides=None, valid=True)


eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": {
            "response": "Paris",
            "retrieved_context": [
                {
                    "content": "Paris is the capital of France.",
                    "source": "wikipedia",
                }
            ],
        },
    },
]

from mlflow.genai.judges import is_grounded
from mlflow.genai.scorers import scorer


@scorer
def is_grounded_scorer(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    return is_grounded(
        request=inputs["query"],
        response=outputs["response"],
        context=outputs["retrieved_context"],
    )


eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=[is_grounded_scorer])

print(eval_results)


eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": {
            "response": "Paris is the magnificent capital city of France, a stunning metropolis known worldwide for its iconic Eiffel Tower, rich cultural heritage, beautiful architecture, world-class museums like the Louvre, and its status as one of Europe's most important political and economic centers. As the capital city, Paris serves as the seat of France's government and is home to numerous important national institutions."
        },
        "expectations": {
            "expected_facts": ["Paris is the capital of France."],
        },
    },
]


from mlflow.genai.scorers import Correctness


eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=[Correctness()])

print(eval_results)

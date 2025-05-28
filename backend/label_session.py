import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import mlflow

# import mlflow


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

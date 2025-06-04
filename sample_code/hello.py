from dotenv import load_dotenv
import mlflow.utils

# import mlflow.utils

# Load environment variables from .env file
load_dotenv()

import mlflow

# from mlflow.genai.judges import meets_guidelines

# # Define your guideline
# tone_guideline = "The response must maintain a courteous, respectful tone throughout. It must show empathy for customer concerns."

# # Call the meets_guidelines API
# feedback = meets_guidelines(
#     guidelines=[tone_guideline],
#     context={
#         "request": "JUST FIX IT FOR ME",
#         "response": "I understand your frustration. Let me help you resolve this immediately.",
#     },
#     name="tone_check",
# )

# # Result
# print(feedback.value)  # "yes"
# print(
#     feedback.rationale
# )  # "The response acknowledges the customer's frustration and offers immediate assistance in a respectful manner."

import mlflow
import os


# @mlflow.trace
# def hello_mlflow(message: str):
#     experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
#     databricks_host = os.environ.get("DATABRICKS_HOST")
#     hello_data = {
#         "experiment_url": f"{databricks_host}/mlflow/experiments/{experiment_id}",
#         "experiment_name": mlflow.get_experiment(experiment_id=experiment_id).name,
#         "message": message,
#     }
#     print(hello_data)
#     return hello_data


# hello_mlflow("hello, world!")


eval_df = pd.DataFrame([
    {
        "inputs": {"inquiry_id": 12926362, "batch_id": 386824328, "grid_entity_id": 239308639},
        "expectations": {"expected_response": "NMH: First and surname match but name is common (about 1,200,000 results).Only YoB provided by RDC and it's 1yr different from client's YoB. Location is over 100 miles. Not enough identifiers to confirm this is a match or consider as possible match with our client.\t"},
        "outputs": "NMH: The alert is classified as a false positive due to the significant location mismatch and the commonality of the name, despite the name and year of birth matching closely.",
    }
])

with mlflow.start_run() as run:
    eval_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(
        df=eval_df,
        name="eval_dataset",
    )   
    # print("experiment_id", experiment_id)
    # print("run_name", run_name)
    # logged_model=mlflow.get_logged_model(model_id="m-89eff9c999e346e89c067dfaae0bb68a")
    # print("logged_model", logged_model)
    # model_id=logged_model.model_id
    # print(f"logged model id: {model_id}")

    
    # from mlflow.genai.scorers import guideline_adherence
    # if log_traces:

        
    english = mlflow.meets_guidelines(
        name="english",
        global_guidelines=["The response must be in English"],
    )
    clarify = meets_guidelines(
        name="clarify",
        global_guidelines=["The response must be clear, coherent, and concise"],
    )

        try:

            mlflow.genai.evaluate(
                model_id=logged_model.model_id, # m-89eff9c999e346e89c067dfaae0bb68a
                data=eval_df,  
                scorers=[
                    english,
                    clarify
                ],
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow evaluation failed: {e}")
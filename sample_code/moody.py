from dotenv import load_dotenv
import mlflow.utils

# import mlflow.utils

# Load environment variables from .env file
load_dotenv()

import mlflow

from mlflow.genai.scorers import GuidelineAdherence
import pandas as pd

eval_df = pd.DataFrame(
    [
        {
            "inputs": {
                "inquiry_id": 12926362,
                "batch_id": 386824328,
                "grid_entity_id": 239308639,
            },
            "expectations": {
                "expected_response": "NMH: First and surname match but name is common (about 1,200,000 results).Only YoB provided by RDC and it's 1yr different from client's YoB. Location is over 100 miles. Not enough identifiers to confirm this is a match or consider as possible match with our client.\t"
            },
            "outputs": "NMH: The alert is classified as a false positive due to the significant location mismatch and the commonality of the name, despite the name and year of birth matching closely.",
        }
    ]
)

with mlflow.start_run() as run:
    mlflow.genai.evaluate(
        # model_id=logged_model.model_id, # m-89eff9c999e346e89c067dfaae0bb68a
        data=eval_df,
        scorers=[
            GuidelineAdherence(
                name="english", global_guidelines=["The response must be in English"]
            ),
            GuidelineAdherence(
                name="clarify",
                global_guidelines=["The response must be clear, coherent, and concise"],
            ),
        ],
    )

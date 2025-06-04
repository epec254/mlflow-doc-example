from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety


# Simple fake chatbot app for demonstration
@mlflow.trace
def my_chatbot_app(question: str) -> dict:
    # Simulate different responses based on the question
    if "MLflow" in question:
        response = "MLflow is an open-source platform for managing machine learning workflows, including tracking experiments, packaging code, and deploying models."
    elif "started" in question:
        response = "To get started with MLflow, install it using 'pip install mlflow' and then run 'mlflow ui' to launch the web interface."
    else:
        response = "I'm a helpful chatbot. Please ask me about MLflow!"

    return {"response": response}


# # Evaluate your app
# results = mlflow.genai.evaluate(
#     data=[
#         {"inputs": {"question": "What is MLflow?"}},
#         {"inputs": {"question": "How do I get started?"}},
#     ],
#     predict_fn=my_chatbot_app,
#     scorers=[RelevanceToQuery(), Safety()],
# )


data = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
)


print(data)

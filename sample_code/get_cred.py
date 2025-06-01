from dotenv import load_dotenv
import mlflow.utils

# Load environment variables from .env file
load_dotenv()
import os
import mlflow

cred = mlflow.utils.databricks_utils.get_databricks_host_creds()
print(cred.token)
print(cred.host)


# mlflow.get_current_active_experiment()

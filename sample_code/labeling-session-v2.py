from dotenv import load_dotenv
import mlflow.genai.label_schemas

# Load environment variables from .env file
load_dotenv()


import mlflow
import mlflow.genai.labeling as labeling
import mlflow.genai.label_schemas as schemas
import pandas as pd
from openai import OpenAI


def test_basic_session_creation():
    """Test: Basic Session Creation"""
    print("=" * 50)
    print("Testing: Basic Session Creation")
    print("=" * 50)

    # Create a simple labeling session with built-in schemas
    session = labeling.create_labeling_session(
        name="customer_service_review_jan_2024",
        assigned_users=["nikhil.thorat@databricks.com", "corey.zumar@databricks.com"],
        label_schemas=[schemas.EXPECTED_FACTS],  # Required: at least one schema needed
    )

    print(f"Created session: {session.name}")
    print(f"Session ID: {session.labeling_session_id}")
    return session


def test_session_with_custom_schemas():
    """Test: Session with Custom Label Schemas"""
    print("=" * 50)
    print("Testing: Session with Custom Label Schemas")
    print("=" * 50)

    # Create custom schemas first (see Labeling Schemas guide)
    quality_schema = schemas.create_label_schema(
        name="response_quality",
        type="feedback",
        title="Rate the response quality",
        input=schemas.InputCategorical(options=["Poor", "Fair", "Good", "Excellent"]),
        overwrite=True,
    )

    # Create session using the schemas
    session = labeling.create_labeling_session(
        name="quality_assessment_session",
        assigned_users=["nikhil.thorat@databricks.com"],
        label_schemas=["response_quality", schemas.EXPECTED_FACTS],
    )
    return session


def test_retrieving_sessions():
    """Test: Retrieving Sessions"""
    print("=" * 50)
    print("Testing: Retrieving Sessions")
    print("=" * 50)

    # Get all labeling sessions
    all_sessions = labeling.get_labeling_sessions()
    print(f"Found {len(all_sessions)} sessions")

    for session in all_sessions:
        print(f"- {session.name} (ID: {session.labeling_session_id})")
        print(f"  Assigned users: {session.assigned_users}")


def test_getting_specific_session():
    """Test: Getting a Specific Session"""
    print("=" * 50)
    print("Testing: Getting a Specific Session")
    print("=" * 50)

    # Get all labeling sessions first
    all_sessions = labeling.get_labeling_sessions()

    # Find session by name (note: names may not be unique)
    target_session = None
    for session in all_sessions:
        if session.name == "customer_service_review_jan_2024":
            target_session = session
            break

    if target_session:
        print(f"Session name: {target_session.name}")
        print(f"Experiment ID: {target_session.experiment_id}")
        print(f"MLflow Run ID: {target_session.mlflow_run_id}")
        print(f"Label schemas: {target_session.label_schemas}")
    else:
        print("Session not found")

    # Alternative: Get session by MLflow Run ID (if you know it)
    # Note: This example won't work without actual run_id and experiment_id
    # run_id = "your_labeling_session_run_id"
    # run = mlflow.search_runs(
    #     experiment_ids=["your_experiment_id"],
    #     filter_string=f"tags.mlflow.runName LIKE '%labeling_session%' AND attribute.run_id = '{run_id}'"
    # ).iloc[0]
    # print(f"Found labeling session run: {run['run_id']}")
    # print(f"Session name: {run['tags.mlflow.runName']}")


def test_deleting_sessions():
    """Test: Deleting Sessions"""
    print("=" * 50)
    print("Testing: Deleting Sessions")
    print("=" * 50)

    # Find the session to delete by name
    all_sessions = labeling.get_labeling_sessions()
    session_to_delete = None
    for session in all_sessions:
        if session.name == "customer_service_review_jan_2024":
            session_to_delete = session
            break

    if session_to_delete:
        # Delete the session (removes from Review App)
        review_app = labeling.delete_labeling_session(session_to_delete)
        print(f"Deleted session: {session_to_delete.name}")
    else:
        print("Session not found")


def test_adding_traces_from_search():
    """Test: Adding Traces from Search Results"""
    print("=" * 50)
    print("Testing: Adding Traces from Search Results")
    print("=" * 50)

    # First, let's create some sample traces with a simple app
    # Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
    # Alternatively, you can use your own OpenAI credentials here
    mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
    client = OpenAI(
        api_key=mlflow_creds.token, base_url=f"{mlflow_creds.host}/serving-endpoints"
    )

    @mlflow.trace
    def support_app(question: str):
        """Simple support app that generates responses"""
        mlflow.update_current_trace(tags={"test_tag": "C001"})
        response = client.chat.completions.create(
            model="databricks-claude-3-7-sonnet",  # This example uses Databricks hosted Claude 3.5 Sonnet. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support agent.",
                },
                {"role": "user", "content": question},
            ],
        )
        return {"response": response.choices[0].message.content}

    # Generate some sample traces
    with mlflow.start_run():
        # Create traces with negative feedback for demonstration
        support_app("My order is delayed")

        support_app("I can't log into my account")

    # Now search for traces to label
    traces_df = mlflow.search_traces(
        filter_string="tags.test_tag = 'C001'", max_results=50
    )

    # Create session and add traces
    session = labeling.create_labeling_session(
        name="negative_feedback_review",
        assigned_users=["nikhil.thorat@databricks.com"],
        label_schemas=["response_quality", schemas.EXPECTED_FACTS],
    )

    # Add traces from search results
    session.add_traces(traces_df)
    print(f"Added {len(traces_df)} traces to session")
    return session


def test_adding_individual_trace_objects():
    """Test: Adding Individual Trace Objects"""
    print("=" * 50)
    print("Testing: Adding Individual Trace Objects")
    print("=" * 50)

    # Set up the app to generate traces
    # Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
    # Alternatively, you can use your own OpenAI credentials here
    mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
    client = OpenAI(
        api_key=mlflow_creds.token, base_url=f"{mlflow_creds.host}/serving-endpoints"
    )

    @mlflow.trace
    def support_app(question: str):
        """Simple support app that generates responses"""
        response = client.chat.completions.create(
            model="databricks-claude-3-7-sonnet",  # This example uses Databricks hosted Claude 3.5 Sonnet. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support agent.",
                },
                {"role": "user", "content": question},
            ],
        )
        return {"response": response.choices[0].message.content}

    # Generate specific traces for edge cases
    with mlflow.start_run() as run:
        # Create traces for specific scenarios
        support_app("What's your refund policy?")
        trace_id_1 = mlflow.get_last_active_trace_id()

        support_app("How do I cancel my subscription?")
        trace_id_2 = mlflow.get_last_active_trace_id()

        support_app("The website is down")
        trace_id_3 = mlflow.get_last_active_trace_id()

    # Get the trace objects
    trace1 = mlflow.get_trace(trace_id_1)
    trace2 = mlflow.get_trace(trace_id_2)
    trace3 = mlflow.get_trace(trace_id_3)

    # Create session and add traces
    session = labeling.create_labeling_session(
        name="negative_feedback_review",
        assigned_users=["nikhil.thorat@databricks.com"],
        label_schemas=["response_quality", schemas.EXPECTED_FACTS],
    )

    # Add individual traces
    session.add_traces([trace1, trace2, trace3])


def test_adding_users():
    """Test: Adding Users to Existing Sessions"""
    print("=" * 50)
    print("Testing: Adding Users to Existing Sessions")
    print("=" * 50)

    # Find existing session by name
    all_sessions = labeling.get_labeling_sessions()
    session = None
    for s in all_sessions:
        if s.name == "customer_review_session":
            session = s
            break

    if session:
        # Add more users to the session
        new_users = ["nikhil.thorat@databricks.com", "corey.zumar@databricks.com"]
        session.set_assigned_users(session.assigned_users + new_users)
        print(f"Session now has users: {session.assigned_users}")
    else:
        print("Session not found")


def test_replacing_users():
    """Test: Replacing Assigned Users"""
    print("=" * 50)
    print("Testing: Replacing Assigned Users")
    print("=" * 50)

    # Find session by name
    all_sessions = labeling.get_labeling_sessions()
    session = None
    for s in all_sessions:
        if s.name == "session_name":
            session = s
            break

    if session:
        # Replace all assigned users
        session.set_assigned_users(
            ["nikhil.thorat@databricks.com", "corey.zumar@databricks.com"]
        )
        print("Updated assigned users list")
    else:
        print("Session not found")


def test_dataset_synchronization():
    """Test: Dataset Synchronization"""
    print("=" * 50)
    print("Testing: Dataset Synchronization")
    print("=" * 50)

    # Find session with completed labels by name
    all_sessions = labeling.get_labeling_sessions()
    session = None
    for s in all_sessions:
        if s.name == "completed_review_session":
            session = s
            break

    if session:
        # Sync expectations to dataset
        session.sync_expectations(dataset_name="customer_service_eval_dataset")
        print("Synced expectations to evaluation dataset")
    else:
        print("Session not found")


def test_best_practices_example():
    """Test: Best Practices Example"""
    print("=" * 50)
    print("Testing: Best Practices Example")
    print("=" * 50)

    # Good: Store run ID for later reference
    # Note: Expanding "..." to make this runnable for testing
    session = labeling.create_labeling_session(
        name="my_session",
        assigned_users=["nikhil.thorat@databricks.com"],
        label_schemas=[schemas.EXPECTED_FACTS],
    )
    session_run_id = session.mlflow_run_id  # Store this!
    print(f"Stored session run ID: {session_run_id}")

    # Later: Use run ID to find session via mlflow.search_runs()
    # rather than searching by name through all sessions


def main():
    """Run all tests"""
    print("Starting MLflow Labeling Sessions Test Suite")
    print("=" * 60)

    facts = schemas.create_label_schema(
        name=schemas.EXPECTED_FACTS,
        type=mlflow.genai.label_schemas.LabelSchemaType.EXPECTATION,
        title="How would you rate the overall quality of this response?",
        input=mlflow.genai.label_schemas.InputCategorical(
            options=["Poor", "Fair", "Good", "Excellent"]
        ),
        instruction="Consider accuracy, relevance, and helpfulness when rating.",
        overwrite=True,
    )

    try:
        # Set up MLflow experiment
        # mlflow.set_experiment("labeling_sessions_test")

        # Run tests in order
        session1 = test_basic_session_creation()
        session2 = test_session_with_custom_schemas()
        test_retrieving_sessions()
        test_getting_specific_session()
        test_deleting_sessions()
        session3 = test_adding_traces_from_search()
        test_adding_individual_trace_objects()
        test_adding_users()
        test_replacing_users()
        test_dataset_synchronization()
        test_best_practices_example()

        print("\n" + "=" * 60)
        print("All tests completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

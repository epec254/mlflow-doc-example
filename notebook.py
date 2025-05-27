# Databricks notebook source
# MAGIC %pip uninstall mlflow mlflow-skinny -y
# MAGIC %pip install https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-studio/dev/samraj.moorjani/databricks_agents-1.0.0rc1.dev0-py3-none-any.whl
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@master -q
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@master -q
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@master#subdirectory=skinny -q
# MAGIC %pip install databricks-sdk[openai]>=0.35.0
# MAGIC %restart_python

# COMMAND ----------

import json

with open("input_data.jsonl", "r") as file:
    email_data = [{"inputs": {"customer_info": json.loads(line)}} for line in file]

# COMMAND ----------

email_data[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's create an email generation prompt
# MAGIC
# MAGIC When the prompt registry is available, instead, you'd register this prompt:
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC prompt = mlflow.register_prompt(
# MAGIC     name="email-prompt",
# MAGIC     template=prompt,
# MAGIC     # Optional: Provide a commit message to describe the changes
# MAGIC     commit_message="Initial commit",
# MAGIC     # Optional: Specify any additional metadata about the prompt version
# MAGIC     version_metadata={
# MAGIC         "author": "eric@example.com",
# MAGIC     },
# MAGIC     # Optional: Set tags applies to the prompt (across versions)
# MAGIC     tags={
# MAGIC         "task": "emails",
# MAGIC         "language": "en",
# MAGIC     },
# MAGIC )
# MAGIC ```

# COMMAND ----------

prompt = """You are an expert sales communication assistant for CloudFlow Inc. Your task is to generate a personalized, professional follow-up email for our sales representatives to send to their customers at the end of the day.

## INPUT DATA
You will be provided with a JSON object containing:
- Account information
- Recent activity data (meetings, product usage, support tickets)
- Sales representative details

## EMAIL REQUIREMENTS
Generate an email that follows these guidelines:

1. SUBJECT LINE:
   - Engaging and attention-grabbing
   - Include the company name if appropriate

2. GREETING:
   - Address the main contact by first name
   - Use a professional but friendly opening

3. BODY CONTENT:
   - Begin with a brief mention of CloudFlow's recent improvements
   - Reference the most recent meeting/interaction
   - Provide updates on support tickets
   - Highlight positive product usage trends
   - Address any action items from previous meetings
   - Include recommendations for additional features they should try
   - Suggest scheduling a follow-up meeting soon

4. TONE AND STYLE:
   - Professional but enthusiastic
   - Show expertise by using industry terminology
   - Include at least one customer success story
   - Balance being informative with driving future business
   - Personalized where possible
   - Ensure email is comprehensive enough to cover all important points

5. CLOSING:
   - Include an appropriate sign-off
   - Use the sales rep's signature from the provided data
   - Add a brief mention of an upcoming product release or feature

## OUTPUT FORMAT
Provide the complete email as JUST a JSON object that can be loaded via `json.loads()` (do not wrap the JSON in backticks) with:
- `subject_line`: Subject line
- `body`: Body content with appropriate spacing and formatting including the signature

Remember, this email should position the sales representative as a trusted advisor who can help the customer get maximum value from CloudFlow's solutions."""

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Let's give it a try

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient
import mlflow

# Turn on tracing
mlflow.openai.autolog()

# Can be ANY model!
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()


@mlflow.trace
def generate_email(customer_info, prompt):
    response = openai_client.chat.completions.create(
        model="agents-demo-gpt4o",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(customer_info),
            },
        ],
    )

    # Parse JSON
    with mlflow.start_span(name="parse_json") as span:
        s = response.choices[0].message.content
        span.set_inputs({"model_output": s})
        clean_string = (
            s[3:-3] if s.startswith("```json\n") and s.endswith("\n```") else s
        )
        span.set_outputs({"json_string": clean_string})

    email_json = json.loads(clean_string)
    return email_json


email = generate_email(email_data[0], prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now, let's define the evaluation criteria we want to use

# COMMAND ----------

import mlflow
from mlflow.genai import scorers
from mlflow.genai.scorers import scorer
from databricks.agents.evals import judges
from mlflow.entities import Feedback

# Check groundedness with the built-in Databricks judge


@scorer
def grounded(inputs, outputs):
    assessment = judges.groundedness(
        request="Write an email for this customer.",
        response=outputs["body"],
        retrieved_context=[{"content": json.dumps(inputs["customer_info"])}],
    )
    return Feedback(
        name="grounded", value=assessment.value, rationale=assessment.rationale
    )


# The Databricks built-in judges can be customized with guidelines
guidelines = {
    "accuracy": """The response correctly references all factual information from the provided_info based on these rules:
- All factual information must be directly sourced from the provided data with NO fabrication
- Names, dates, numbers, and company details must be 100% accurate with no errors
- Meeting discussions must be summarized with the exact same sentiment and priority as presented in the data
- Support ticket information must include correct ticket IDs, status, and resolution details when available
- All product usage statistics must be presented with the same metrics provided in the data
- No references to CloudFlow features, services, or offerings unless specifically mentioned in the customer data
- AUTOMATIC FAIL if any information is mentioned that is not explicitly provided in the data""",
    "personalized": """The response demonstrates clear personalization based on the provided_info based on these rules:
- Email must begin by referencing the most recent meeting/interaction
- Immediatly next, the email must address the customer's MOST pressing concern as evidenced in the data
- Content structure must be customized based on the account's health status (critical issues first for "Fair" or "Poor" accounts)
- Industry-specific language must be used that reflects the customer's sector
- Recommendations must ONLY reference features that are:
  a) Listed as "least_used_features" in the data, AND
  b) Directly related to the "potential_opportunity" field
- Relationship history must be acknowledged (new vs. mature relationship)
- Deal stage must influence communication approach (implementation vs. renewal vs. growth)
- AUTOMATIC FAIL if recommendations could be copied to another customer in a different situation""",
    "relevance": """The response prioritizes content that matters to the recipient in the provided_info based on these rules:
- Critical support tickets (status="Open (Critical)") must be addressed after the greeting, reference to the most recent interaction, any pleasantrys, and references to closed tickets
    - it is ok if they name is slightly different as long as it is clearly the same issue as in the provided_info
- Time-sensitive action items must be addressed before general updates
- Content must be ordered by descending urgency as defined by:
  1. Critical support issues
  2. Action items explicitly stated in most recent meeting
  3. Upcoming renewal if within 30 days
  4. Recently resolved issues
  5. Usage trends and recommendations
- No more than ONE feature recommendation for accounts with open critical issues
- No mentions of company news, product releases, or success stories not directly requested by the customer
- No calls to action unrelated to the immediate needs in the data
- AUTOMATIC FAIL if the email requests a meeting without being tied to a specific action item or opportunity in the data""",
}


@scorer
def email_guidelines(inputs, outputs):
    results = []
    for guideline_name, guideline in guidelines.items():
        # output = judges.guideline_adherence(request=json.dumps(inputs['customer_info']), guidelines=[guideline], response=outputs['email_text'])
        output = judges.guideline_adherence(
            request="Write an email for this customer.",
            guidelines=[guideline],
            response=outputs["body"],
            guidelines_context={"provided_info": json.dumps(inputs["customer_info"])},
        )
        results.append(
            Feedback(
                name=guideline_name, value=output.value, rationale=output.rationale
            )
        )
    return results


# You can define a custom evaluation metric using a Python function
@scorer
def rep_name_in_email(inputs, outputs):
    return inputs["customer_info"]["sales_rep"]["name"] in outputs["body"]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Now let's evaluate the prompt over a larger set of emails!

# COMMAND ----------


# Helper function to use the prompt
@mlflow.trace
def generate_email_v1(customer_info):
    return generate_email(customer_info, prompt)


# MLflow's evaluation harness runs your app against any data inputs
mlflow.genai.evaluate(
    data=email_data[:10],
    predict_fn=generate_email_v1,
    scorers=[scorers.safety(), email_guidelines, grounded, rep_name_in_email],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's iterate on the prompt...
# MAGIC
# MAGIC In the future, you'd update the prompt in the registry.

# COMMAND ----------

prompt_v2 = """
You are an expert sales communication assistant for CloudFlow Inc. Your task is to generate a personalized, professional follow-up email for our sales representatives to send to their customers at the end of the day.

## INPUT DATA
You will be provided with a JSON object containing:
- Account information
- Recent activity data (meetings, product usage, support tickets)
- Sales representative details

## EMAIL REQUIREMENTS
Generate an email that follows these guidelines:

1. SUBJECT LINE:
   - Concise and specific to the most important update or follow-up point
   - Include the company name if appropriate

2. GREETING:
   - Address the main contact by first name
   - Use a professional but friendly opening

3. BODY CONTENT (prioritize in this order):
   - Reference the most recent meeting/interaction and acknowledge key points discussed
   - Discuss any pressing issues that are still open immediatly afterwards
   - Provide updates on any urgent or recently resolved support tickets
   - Highlight positive product usage trends or achievements
   - Address any specific action items from previous meetings
   - Include personalized recommendations based on features listed as 'least_used_features' and directly related to the 'potential_opportunity' field.  
      - Make sure these recommendations can NOT be copied to another customer in a different situation
      - No more than ONE feature recommendation for accounts with open critical issues
   - Suggest clear and specific next steps
      - Only request a meeting if it can be tied to specific action items
      

4. TONE AND STYLE:
   - Professional but conversational
   - Concise paragraphs (2-3 sentences each)
   - Use bullet points for lists or multiple items
   - Balance between being informative and actionable
   - Personalized to reflect the existing relationship
   - Adjust formality based on the customer's industry and relationship history

5. CLOSING:
   - Include an appropriate sign-off
   - Use the sales rep's signature from the provided data
   - No generic marketing language or overly sales-focused calls to action

## OUTPUT FORMAT
Provide the complete email as JUST a JSON object that can be loaded via `json.loads()` (do not wrap the JSON in backticks) with:
- `subject_line`: Subject line
- `body`: Body content with appropriate spacing and formatting including the signature

Remember, this email should feel like it was thoughtfully written by the sales representative based on their specific knowledge of the customer, not like an automated message."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## And evaluate again...

# COMMAND ----------


# Helper function to use the prompt
@mlflow.trace
def generate_email_v2(customer_info):
    return generate_email(customer_info, prompt_v2)


# MLflow's evaluation harness runs your app against any data inputs
mlflow.genai.evaluate(
    data=email_data[:10],
    predict_fn=generate_email_v2,
    scorers=[scorers.safety(), email_guidelines, grounded, rep_name_in_email],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Things look much better, so its time to get feedback from domain experts

# COMMAND ----------

# Helper function to use the prompt
from mlflow.entities import Document


@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(customer_info):
    return [
        Document(
            id=customer_info["inputs"]["customer_info"]["account"]["name"],
            page_content=f"```markdown\n{json.dumps(customer_info['inputs']['customer_info'], indent=2)}\n```",
            metadata={"doc_uri": "customer_info"},
        )
    ]


# @mlflow.trace
def generate_email_test(customer_info):
    with mlflow.start_span() as span:
        span.set_inputs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": customer_info["inputs"]["customer_info"]["account"][
                            "name"
                        ],
                    }
                ]
            }
        )
        retrieve_docs(customer_info)
        output = generate_email(customer_info["inputs"]["customer_info"], prompt_v2)
        span.set_outputs(
            [
                {
                    "role": "assistant",
                    "content": f"## Subject\n{output['subject_line']}\n\n---------\n\n{output['body']}",
                }
            ]
        )


with mlflow.start_run(run_name="examples"):
    for email in email_data[:10]:
        generate_email_test(email)

# COMMAND ----------

import mlflow
from databricks.agents import review_app
from databricks.agents import datasets

# The review app is tied to the current MLFlow experiment.
my_app = review_app.get_review_app()

# Search for the traces from the above run
traces = mlflow.search_traces(run_id="54bf2b01427d4a2d9cec1c347197468f")

accuracy_label_schema = my_app.create_label_schema(
    name="accuracy",
    # Type can be "expectation" or "feedback".
    type="feedback",
    title="Is the email accurate?",
    # see docs for other question formats: https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#label-schemas
    input=review_app.label_schemas.InputCategorical(options=["Yes", "Partially", "No"]),
    instruction="Determine if the email correctly references all factual information from the provided info.",
    enable_comment=True,
    overwrite=True,
)

personalized_label_schema = my_app.create_label_schema(
    name="personalized",
    # Type can be "expectation" or "feedback".
    type="feedback",
    title="Is the email personalized?",
    # see docs for other question formats: https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#label-schemas
    input=review_app.label_schemas.InputCategorical(options=["Yes", "Partially", "No"]),
    instruction="Determine if the email is sufficiently personalized based on the provided info.",
    enable_comment=True,
    overwrite=True,
)

relevant_label_schema = my_app.create_label_schema(
    name="relevance",
    # Type can be "expectation" or "feedback".
    type="feedback",
    title="Is the email relevant?",
    # see docs for other question formats: https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#label-schemas
    input=review_app.label_schemas.InputCategorical(options=["Yes", "Partially", "No"]),
    instruction="Determine if the email is relevant, that is, it prioritizes content that matters to the recipient based on the provided info.",
    enable_comment=True,
    overwrite=True,
)


my_session = my_app.create_labeling_session(
    name="label_emails",
    assigned_users=[],
    label_schemas=["relevance", "personalized", "accuracy"],
)
# NOTE: This will copy the traces into this labeling session so that labels do not modify the original traces.
my_session.add_traces(traces)
print(my_session.url)

# COMMAND ----------

# MAGIC %md
# MAGIC Topics for future sessions
# MAGIC
# MAGIC - Evaluation Datasets
# MAGIC - Logging agents to MLflow
# MAGIC - Deploying agents as endpoints
# MAGIC - Monitoring

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import mlflow


# Built-in schemas access
print("=== Built-in Schemas ===")
import mlflow.genai.label_schemas as schemas


# schemas.get_label_schema("test")


# Built-in schema names
guidelines_name = schemas.GUIDELINES
expected_facts_name = schemas.EXPECTED_FACTS
expected_response_name = schemas.EXPECTED_RESPONSE

print(f"Guidelines schema name: {guidelines_name}")
print(f"Expected facts schema name: {expected_facts_name}")
print(f"Expected response schema name: {expected_response_name}")

# Basic Schema Creation
print("\n=== Basic Schema Creation ===")
import mlflow.genai.label_schemas as schemas
from mlflow.genai.label_schemas import InputCategorical, InputText

# Create a feedback schema for rating response quality
quality_schema = schemas.create_label_schema(
    name="response_quality",
    type="feedback",
    title="How would you rate the overall quality of this response?",
    input=InputCategorical(options=["Poor", "Fair", "Good", "Excellent"]),
    instruction="Consider accuracy, relevance, and helpfulness when rating.",
)
print(f"Created schema: {quality_schema.name}")

# Schema Types
print("\n=== Schema Types ===")
from mlflow.genai.label_schemas import InputTextList

# Feedback schema for subjective assessment
tone_schema = schemas.create_label_schema(
    name="response_tone",
    type="feedback",
    title="Is the response tone appropriate for the context?",
    input=InputCategorical(options=["Too formal", "Just right", "Too casual"]),
    enable_comment=True,  # Allow additional comments
)

# Expectation schema for ground truth
facts_schema = schemas.create_label_schema(
    name="required_facts",
    type="expectation",
    title="What facts must be included in a correct response?",
    input=InputTextList(max_count=5, max_length_each=200),
    instruction="List key facts that any correct response must contain.",
)

print(f"Created feedback schema: {tone_schema.name}")
print(f"Created expectation schema: {facts_schema.name}")

# Retrieving Schemas
print("\n=== Retrieving Schemas ===")

# Get an existing schema
schema = schemas.get_label_schema("response_quality")
print(f"Schema: {schema.name}")
print(f"Type: {schema.type}")
print(f"Title: {schema.title}")

# Updating Schemas
print("\n=== Updating Schemas ===")

# Update by recreating with overwrite=True
updated_schema = schemas.create_label_schema(
    name="response_quality",
    type="feedback",
    title="Rate the response quality (updated question)",
    input=InputCategorical(options=["Excellent", "Good", "Fair", "Poor", "Very Poor"]),
    instruction="Updated: Focus on factual accuracy above all else.",
    overwrite=True,  # Replace existing schema
)
print(f"Updated schema: {updated_schema.title}")

# Input Types Examples
print("\n=== Input Types Examples ===")

# Single-Select Dropdown (InputCategorical)
print("InputCategorical examples:")

# Rating scale
rating_input = InputCategorical(
    options=[
        "1 - Poor",
        "2 - Below Average",
        "3 - Average",
        "4 - Good",
        "5 - Excellent",
    ]
)

# Binary choice
safety_input = InputCategorical(options=["Safe", "Unsafe"])

# Multiple categories
error_type_input = InputCategorical(
    options=["Factual Error", "Logical Error", "Formatting Error", "No Error"]
)

print(f"Rating input options: {rating_input.options}")
print(f"Safety input options: {safety_input.options}")
print(f"Error type input options: {error_type_input.options}")

# Multi-Select Dropdown (InputCategoricalList)
print("\nInputCategoricalList examples:")
from mlflow.genai.label_schemas import InputCategoricalList

# Multiple error types can be present
errors_input = InputCategoricalList(
    options=[
        "Factual inaccuracy",
        "Missing context",
        "Inappropriate tone",
        "Formatting issues",
        "Off-topic content",
    ]
)

# Multiple content types
content_input = InputCategoricalList(
    options=["Technical details", "Examples", "References", "Code samples"]
)

print(f"Errors input options: {errors_input.options}")
print(f"Content input options: {content_input.options}")

# Free-Form Text (InputText)
print("\nInputText examples:")

# General feedback
feedback_input = InputText(max_length=500)

# Specific improvement suggestions
improvement_input = InputText(max_length=200)  # Limit length for focused feedback

# Short answers
summary_input = InputText(max_length=100)

print(f"Feedback input max length: {feedback_input.max_length}")
print(f"Improvement input max length: {improvement_input.max_length}")
print(f"Summary input max length: {summary_input.max_length}")

# Multiple Text Entries (InputTextList)
print("\nInputTextList examples:")

# List of factual errors
errors_input = InputTextList(
    max_count=10,  # Maximum 10 errors
    max_length_each=150,  # Each error description limited to 150 chars
)

# Missing information
missing_input = InputTextList(max_count=5, max_length_each=200)

# Improvement suggestions
suggestions_input = InputTextList(max_count=3)  # No length limit per item

print(
    f"Errors input - max count: {errors_input.max_count}, max length each: {errors_input.max_length_each}"
)
print(
    f"Missing input - max count: {missing_input.max_count}, max length each: {missing_input.max_length_each}"
)
print(f"Suggestions input - max count: {suggestions_input.max_count}")

# Numeric Input (InputNumeric)
print("\nInputNumeric examples:")
from mlflow.genai.label_schemas import InputNumeric

# Confidence score
confidence_input = InputNumeric(min_value=0.0, max_value=1.0)

# Rating scale
rating_input = InputNumeric(min_value=1, max_value=10)

# Cost estimate
cost_input = InputNumeric(min_value=0)  # No maximum limit

print(
    f"Confidence input range: {confidence_input.min_value} - {confidence_input.max_value}"
)
print(f"Rating input range: {rating_input.min_value} - {rating_input.max_value}")
print(f"Cost input min: {cost_input.min_value}, max: {cost_input.max_value}")

# Complete Examples - Customer Service Evaluation
print("\n=== Complete Example: Customer Service Evaluation ===")

# Overall quality rating
quality_schema = schemas.create_label_schema(
    name="service_quality",
    type="feedback",
    title="Rate the overall quality of this customer service response",
    input=InputCategorical(
        options=["Excellent", "Good", "Average", "Poor", "Very Poor"]
    ),
    instruction="Consider helpfulness, accuracy, and professionalism.",
    enable_comment=True,
)

# Issues identification
issues_schema = schemas.create_label_schema(
    name="response_issues",
    type="feedback",
    title="What issues are present in this response? (Select all that apply)",
    input=InputCategoricalList(
        options=[
            "Factually incorrect information",
            "Unprofessional tone",
            "Doesn't address the question",
            "Too vague or generic",
            "Contains harmful content",
            "No issues identified",
        ]
    ),
    instruction="Select all issues you identify. Choose 'No issues identified' if the response is problem-free.",
)

# Expected resolution steps
resolution_schema = schemas.create_label_schema(
    name="expected_resolution",
    type="expectation",
    title="What steps should be included in the ideal resolution?",
    input=InputTextList(max_count=5, max_length_each=200),
    instruction="List the key steps a customer service rep should take to properly resolve this issue.",
)

# Confidence in assessment
confidence_schema = schemas.create_label_schema(
    name="assessment_confidence",
    type="feedback",
    title="How confident are you in your assessment?",
    input=InputNumeric(min_value=1, max_value=10),
    instruction="Rate from 1 (not confident) to 10 (very confident)",
)

print(f"Created customer service schemas:")
print(f"  - {quality_schema.name}: {quality_schema.title}")
print(f"  - {issues_schema.name}: {issues_schema.title}")
print(f"  - {resolution_schema.name}: {resolution_schema.title}")
print(f"  - {confidence_schema.name}: {confidence_schema.title}")

# Complete Examples - Medical Information Review
print("\n=== Complete Example: Medical Information Review ===")

# Safety assessment
safety_schema = schemas.create_label_schema(
    name="medical_safety",
    type="feedback",
    title="Is this medical information safe and appropriate?",
    input=InputCategorical(
        options=[
            "Safe - appropriate general information",
            "Concerning - may mislead patients",
            "Dangerous - could cause harm if followed",
        ]
    ),
    instruction="Assess whether the information could be safely consumed by patients.",
)

# Required disclaimers
disclaimers_schema = schemas.create_label_schema(
    name="required_disclaimers",
    type="expectation",
    title="What medical disclaimers should be included?",
    input=InputTextList(max_count=3, max_length_each=300),
    instruction="List disclaimers that should be present (e.g., 'consult your doctor', 'not professional medical advice').",
)

# Accuracy of medical facts
accuracy_schema = schemas.create_label_schema(
    name="medical_accuracy",
    type="feedback",
    title="Rate the factual accuracy of the medical information",
    input=InputNumeric(min_value=0, max_value=100),
    instruction="Score from 0 (completely inaccurate) to 100 (completely accurate)",
)

print(f"Created medical information schemas:")
print(f"  - {safety_schema.name}: {safety_schema.title}")
print(f"  - {disclaimers_schema.name}: {disclaimers_schema.title}")
print(f"  - {accuracy_schema.name}: {accuracy_schema.title}")

# Integration with Labeling Sessions
print("\n=== Integration with Labeling Sessions ===")

# Schemas are automatically available when creating labeling sessions
# The Review App will present questions based on your schema definitions

# Example: Using schemas in a session (conceptual - actual session creation
# happens through the Review App UI or other APIs)
session_schemas = [
    "service_quality",  # Your custom schema
    "response_issues",  # Your custom schema
    schemas.EXPECTED_FACTS,  # Built-in schema
]

print(f"Session schemas: {session_schemas}")

# Test deleting a schema
print("\n=== Testing Schema Deletion ===")
try:
    # Create a test schema to delete
    test_schema = schemas.create_label_schema(
        name="test_to_delete",
        type="feedback",
        title="This is a test schema",
        input=InputText(max_length=100),
    )
    print(f"Created test schema: {test_schema.name}")

    # Delete it
    schemas.delete_label_schema("test_to_delete")
    print("Successfully deleted test schema")

except Exception as e:
    print(f"Error during deletion test: {e}")

print("\n=== All tests completed! ===")

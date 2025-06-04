"""
ESSENTIAL MLflow FEEDBACK AND EXPECTATION LOGGING PATTERNS

This script demonstrates the key unique usage patterns for mlflow.log_feedback() and mlflow.log_expectation().
Each example shows a distinct API usage pattern, not just variations of the same approach.

KEY DIFFERENCES:
================
- LOG_FEEDBACK: Restricted to basic types (number, boolean, string, list of strings)
- LOG_EXPECTATION: Supports any JSON-serializable data including complex structures

UNIQUE PATTERNS DEMONSTRATED:
============================
1. Basic feedback types and source variations
2. Error handling for failed evaluations
3. Complex structured expectations vs simple feedback
4. Metadata usage patterns
"""

from dotenv import load_dotenv
import mlflow.utils

# Load environment variables from .env file
load_dotenv()

import os

os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

import mlflow
from mlflow.entities.assessment import (
    AssessmentSource,
    AssessmentSourceType,
    AssessmentError,
)


@mlflow.trace
def my_app(input: str) -> str:
    return input + "_output"


my_app(input="hello")

trace_id = mlflow.get_last_active_trace_id()

# Handle case where trace_id might be None
if trace_id is None:
    raise ValueError("No active trace found. Make sure to run a traced function first.")

print(f"Using trace_id: {trace_id}")

# =============================================================================
# LOG_FEEDBACK - Key Unique Patterns
# =============================================================================


mlflow.log_feedback(
    trace_id=trace_id,
    name="human_rating",
    value=4,  # int
    rationale="Human evaluator rating",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="evaluator@company.com",
    ),
)

mlflow.log_feedback(
    trace_id=trace_id,
    name="llm_judge_score",
    value=0.85,  # float
    rationale="LLM judge evaluation",
    source=AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id="gpt-4o-mini",
    ),
    metadata={"temperature": "0.1", "model_version": "2024-01"},
)

mlflow.log_feedback(
    trace_id=trace_id,
    name="is_helpful",
    value=True,  # bool
    rationale="Boolean assessment of helpfulness",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="reviewer@company.com",
    ),
)

mlflow.log_feedback(
    trace_id=trace_id,
    name="automated_categories",
    value=["helpful", "accurate", "concise"],  # list of strings
    rationale="Automated categorization",
    source=AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="classifier_v1.2",
    ),
)

# For structured data, use log_expectation or put data in metadata/rationale
mlflow.log_feedback(
    trace_id=trace_id,
    name="response_analysis_score",
    value=4.2,  # single score instead of dict
    rationale="Analysis: 150 words, positive sentiment, includes examples, confidence 0.92",
    source=AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="analyzer_v2.1",
    ),
    metadata={
        "word_count": "150",
        "sentiment": "positive",
        "has_examples": "true",
        "confidence": "0.92",
    },
)

# Pattern 2: Error handling when evaluation fails
mlflow.log_feedback(
    trace_id=trace_id,
    name="failed_evaluation",
    source=AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id="gpt-4o",
    ),
    error=AssessmentError(
        error_code="RATE_LIMIT_EXCEEDED",
        error_message="API rate limit exceeded during evaluation",
    ),
    metadata={"retry_count": "3", "error_timestamp": "2024-01-15T10:30:00Z"},
)

# =============================================================================
# LOG_EXPECTATION - Key Unique Patterns
# =============================================================================

# Pattern 1: Simple expectation (string/basic type)
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_response",
    value="The capital of France is Paris.",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="content_curator@example.com",
    ),
)

# Pattern 2: Complex structured expectation (showcases flexibility)
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_response_structure",
    value={
        "entities": {
            "people": ["Marie Curie", "Pierre Curie"],
            "locations": ["Paris", "France"],
            "dates": ["1867", "1934"],
        },
        "key_facts": [
            "First woman to win Nobel Prize",
            "Won Nobel Prizes in Physics and Chemistry",
            "Discovered radium and polonium",
        ],
        "response_requirements": {
            "tone": "informative",
            "length_range": {"min": 100, "max": 300},
            "include_examples": True,
            "citations_required": False,
        },
    },
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="content_strategist@example.com",
    ),
    metadata={
        "content_type": "biographical_summary",
        "target_audience": "general_public",
        "fact_check_date": "2024-01-15",
    },
)

# Pattern 3: List of expected facts
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_facts",
    value=[
        "Paris is the capital of France",
        "The capital city of France is Paris",
        "France's capital is Paris",
    ],
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="qa_team@example.com",
    ),
)

print("Essential feedback and expectation patterns demonstrated successfully!")


import mlflow


@mlflow.trace
def my_app(input: str) -> str:
    return input + "_output"


my_app(input="hello")

trace_id = mlflow.get_last_active_trace_id()


# Log a thumbs up/down rating
mlflow.log_feedback(
    trace_id=trace_id,
    name="quality_rating",
    value=1,  # 1 for thumbs up, 0 for thumbs down
    rationale="The response was accurate and helpful",
    source=mlflow.entities.assessment.AssessmentSource(
        source_type=mlflow.entities.assessment.AssessmentSourceType.HUMAN,
        source_id="bob@example.com",
    ),
)

# Log expected response text
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_response",
    value="The capital of France is Paris.",
    source=mlflow.entities.assessment.AssessmentSource(
        source_type=mlflow.entities.assessment.AssessmentSourceType.HUMAN,
        source_id="bob@example.com",
    ),
)

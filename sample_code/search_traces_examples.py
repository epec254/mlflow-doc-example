"""
MLFLOW SEARCH_TRACES API EXAMPLES

This file demonstrates valid search_traces filter patterns based on the backend implementation.
All examples are verified against the Scala backend parser code.

SUPPORTED FILTER SYNTAX:
=======================
- Tables: traces/trace, attributes/attribute, tags/tag, metadata/request_metadata
- String operators: =, != (for status, name, tags)
- Numeric operators: =, <, <=, >, >= (for timestamps, execution_time)
- Logical: AND (OR is not supported)
- Only single binary expressions or AND conjunctions are supported

SUPPORTED COLUMNS:
=================
- traces.status (string)
- traces.name (string)
- traces.timestamp_ms (numeric) + aliases: timestamp, timestampMs, created
- traces.execution_time_ms (numeric) + aliases: executionTimeMs, executionTime, latency, execution_time
- tags.<key> (string)
- metadata.<key> (string)
"""

from dotenv import load_dotenv
import mlflow
import os

# Load environment variables from .env file
load_dotenv()

os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"


@mlflow.trace
def my_app(input: str) -> str:
    return input + "_output"


# Create a trace to search for
my_app(input="hello")

trace_id = mlflow.get_last_active_trace_id()

# Handle case where trace_id might be None
if trace_id is None:
    print("No active trace found. Creating sample traces...")
    # Run multiple traced functions to create searchable traces
    for i in range(3):
        my_app(input=f"test_{i}")

print("=== MLFLOW SEARCH_TRACES EXAMPLES ===\n")

# =============================================================================
# 1. BASIC ATTRIBUTE FILTERING
# =============================================================================

print("1. Search by trace status:")
traces = mlflow.search_traces(filter_string="traces.status = 'OK'")
print(f"   Found {len(traces)} traces with OK status\n")

print("2. Search by trace name:")
traces = mlflow.search_traces(filter_string="traces.name = 'my_app'")
print(f"   Found {len(traces)} traces with name 'my_app'\n")

# =============================================================================
# 2. TIMESTAMP FILTERING (with aliases)
# =============================================================================

print("3. Search by timestamp (recent traces):")
# Using timestamp in milliseconds (Unix epoch * 1000)
recent_timestamp = 1700000000000  # Adjust this to a relevant timestamp
traces = mlflow.search_traces(filter_string=f"traces.timestamp_ms > {recent_timestamp}")
print(f"   Found {len(traces)} recent traces\n")

print("4. Timestamp aliases:")
# These are equivalent ways to filter by timestamp
filters = [
    f"traces.timestamp > {recent_timestamp}",
    f"traces.timestampMs > {recent_timestamp}",
    f"traces.created > {recent_timestamp}",
]
for filter_str in filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 3. EXECUTION TIME FILTERING (with aliases)
# =============================================================================

print("5. Search by execution time:")
traces = mlflow.search_traces(filter_string="traces.execution_time_ms > 100")
print(f"   Found {len(traces)} traces taking more than 100ms\n")

print("6. Execution time aliases:")
# These are equivalent ways to filter by execution time
exec_time_filters = [
    "traces.executionTimeMs > 50",
    "traces.executionTime > 50",
    "traces.latency > 50",
    "traces.execution_time > 50",
]
for filter_str in exec_time_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 4. TAG FILTERING
# =============================================================================

print("7. Search by tags:")
# Note: These will likely return 0 results unless you've set these tags
tag_filters = [
    "tags.environment = 'production'",
    "tags.version = 'v1.0'",
    "tags.user = 'alice'",
    "tags.experiment_name = 'my_experiment'",
]
for filter_str in tag_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 5. METADATA FILTERING
# =============================================================================

print("8. Search by request metadata:")
# Note: These will likely return 0 results unless you've set this metadata
metadata_filters = [
    "metadata.user_id = 'user123'",
    "metadata.session_id = 'session_abc'",
    "request_metadata.client_version = '1.0.0'",
]
for filter_str in metadata_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 6. COMPLEX FILTERS WITH AND
# =============================================================================

print("9. Complex filters with AND:")
complex_filters = [
    "traces.status = 'OK' AND traces.execution_time_ms < 1000",
    "traces.name = 'my_app' AND traces.timestamp_ms > 1700000000000",
    "traces.status = 'OK' AND tags.environment = 'production'",
    "traces.execution_time_ms > 100 AND traces.execution_time_ms < 1000",
]
for filter_str in complex_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 7. ORDERING EXAMPLES
# =============================================================================

print("10. Search with ordering:")

# Order by timestamp (most recent first)
traces = mlflow.search_traces(
    filter_string="traces.status = 'OK'", order_by=["traces.timestamp_ms DESC"]
)
print(f"    Recent successful traces: {len(traces)} traces")

# Order by execution time (fastest first)
traces = mlflow.search_traces(
    filter_string="traces.status = 'OK'", order_by=["traces.execution_time_ms ASC"]
)
print(f"    Fastest successful traces: {len(traces)} traces")

# Multiple order criteria
traces = mlflow.search_traces(
    order_by=["traces.timestamp_ms DESC", "traces.execution_time_ms ASC"]
)
print(f"    Ordered by time DESC, then execution time ASC: {len(traces)} traces")
print()

# =============================================================================
# 8. NUMERIC COMPARISON OPERATORS
# =============================================================================

print("11. Numeric comparison operators:")
numeric_filters = [
    "traces.execution_time_ms = 100",  # exact match
    "traces.execution_time_ms < 100",  # less than
    "traces.execution_time_ms <= 100",  # less than or equal
    "traces.execution_time_ms > 100",  # greater than
    "traces.execution_time_ms >= 100",  # greater than or equal
    "traces.timestamp_ms > 1700000000000",  # timestamp comparisons
]
for filter_str in numeric_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

# =============================================================================
# 9. STRING COMPARISON OPERATORS
# =============================================================================

print("12. String comparison operators:")
string_filters = [
    "traces.status = 'OK'",  # equality
    "traces.status != 'ERROR'",  # inequality
    "traces.name = 'my_app'",  # name equality
    "traces.name != 'other_app'",  # name inequality
    "tags.environment = 'prod'",  # tag equality
    "tags.environment != 'dev'",  # tag inequality
]
for filter_str in string_filters:
    traces = mlflow.search_traces(filter_string=filter_str)
    print(f"   {filter_str}: {len(traces)} traces")
print()

print("=== SEARCH_TRACES EXAMPLES COMPLETED ===")
print("\nNOTE: Many filters may return 0 results if the corresponding")
print("tags/metadata haven't been set on your traces. This is expected.")
print("\nTo see more results, try:")
print("1. Running more traced functions")
print("2. Setting custom tags: mlflow.set_tag('key', 'value')")
print("3. Using filters that match your actual trace data")

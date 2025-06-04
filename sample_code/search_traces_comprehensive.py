"""
# MLflow Search Traces API - Comprehensive Guide

## Overview

The `mlflow.search_traces()` API allows you to search through traces based on their metadata,
tags, and core attributes. The search operates over the trace JSON structure and supports
filtering and ordering with specific syntax requirements.

## Trace Structure

Each trace contains the following searchable sections:

```json
{
    "trace": {
        "trace_info": {
            "trace_id": "tr-...",
            "state": "ERROR",  // -> attributes.status
            "request_time": "2025-06-04T00:17:06.901Z",  // -> attributes.timestamp_ms
            "execution_duration": "1.840s",  // -> attributes.execution_time_ms
            "trace_metadata": {  // -> metadata.*
                "mlflow.traceOutputs": "response content",
                "mlflow.traceInputs": "request content",
                "mlflow.user": "eric.peter",
                "mlflow.source.name": "runner.py"
            },
            "tags": {  // -> tags.*
                "mlflow.traceName": "process_chat_request",
                "mlflow.user": "7422288164188866"
            }
        }
    }
}
```

## Searchable Fields

### 1. **Attributes** (Core trace properties)
- `attributes.status` - Trace state (OK, ERROR, etc.)
- `attributes.name` - Trace name (rarely used)
- `attributes.timestamp_ms` - Creation time in milliseconds since Unix epoch
- `attributes.execution_time_ms` - Execution duration in milliseconds

**Aliases supported:**
- `timestamp`, `timestampMs`, `created` → `timestamp_ms`
- `executionTimeMs`, `executionTime`, `latency`, `execution_time` → `execution_time_ms`

### 2. **Metadata** (trace_metadata fields)
- `metadata.*` - Any field from trace_metadata
- Common fields: `mlflow.traceOutputs`, `mlflow.traceInputs`, `mlflow.user`, `mlflow.source.name`

### 3. **Tags** (Custom and MLflow tags)
- `tags.*` - Any tag set on the trace
- Common tags: `mlflow.traceName`, `mlflow.user`

## Supported Operators

### String Fields (status, name, tags, metadata)
- `=` (equality)
- `!=` (inequality)

### Numeric Fields (timestamp_ms, execution_time_ms)
- `=`, `<`, `<=`, `>`, `>=`

### Logical Operators
- `AND` (supported)
- `OR` (NOT supported)

## Search Syntax Rules

1. **Table prefixes required**: `attributes.`, `tags.`, `metadata.`
2. **Quoted field names**: Use backticks for fields with special characters: `metadata.\`mlflow.user\``
3. **Single expressions or AND only**: Complex nested logic not supported
4. **Case sensitive**: Field names and values are case sensitive

---

## Practical Examples

"""

from dotenv import load_dotenv
import mlflow
import os
import time

# Load environment variables from .env file
load_dotenv()

os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"


def safe_len(traces):
    """Helper function to get length of traces regardless of type"""
    return len(traces) if traces is not None else 0


def safe_get_first_trace(traces):
    """Helper function to safely get first trace from either DataFrame or list"""
    if traces is None or len(traces) == 0:
        return None
    if hasattr(traces, "iloc"):
        # It's a DataFrame
        return traces.iloc[0]
    elif isinstance(traces, list):
        # It's a list
        return traces[0]
    else:
        return None


@mlflow.trace
def process_chat_request(messages, customer_id, session_id):
    """Example traced function that simulates a chat processing system"""
    # Simulate some processing time
    time.sleep(0.1)

    # Set custom tags using update_current_trace
    mlflow.update_current_trace(
        tags={
            "customer_id": customer_id,
            "session_type": "chat",
            "environment": "production",
        }
    )

    response = f"Processed {len(messages)} messages for customer {customer_id}"
    return response


@mlflow.trace
def analyze_sentiment(text):
    """Another traced function for sentiment analysis"""
    time.sleep(0.05)

    # Set custom tags using update_current_trace
    mlflow.update_current_trace(
        tags={"analysis_type": "sentiment", "text_length": str(len(text))}
    )

    return "positive" if "good" in text.lower() else "neutral"


# Create sample traces with different characteristics
print("Creating sample traces...")

# Successful chat processing
process_chat_request(
    messages=[{"role": "user", "content": "Hello, I need help"}],
    customer_id="C001",
    session_id="session_123",
)

# Sentiment analysis
analyze_sentiment("This is a good product!")

# Another chat with different customer
process_chat_request(
    messages=[{"role": "user", "content": "Technical support needed"}],
    customer_id="C002",
    session_id="session_456",
)

print("\n" + "=" * 80)
print("COMPREHENSIVE SEARCH_TRACES EXAMPLES")
print("=" * 80)

# =============================================================================
# 1. ATTRIBUTE FILTERING - Core trace properties
# =============================================================================

print("\n## 1. ATTRIBUTE FILTERING")
print("-" * 40)

print("\n### Search by trace status (engineer's examples):")
traces = mlflow.search_traces(filter_string="attributes.status = 'OK'")
print(f"✓ Successful traces: {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="attributes.status = 'ERROR'")
print(f"✗ Failed traces: {safe_len(traces)}")

print("\n### Search by trace name (legacy - rarely used):")
traces = mlflow.search_traces(filter_string="attributes.name = 'foo'")
print(f"Named traces 'foo': {safe_len(traces)}")

print("\n### Search by timestamp (engineer's example with milliseconds):")
# Engineer's example using specific timestamp
traces = mlflow.search_traces(filter_string="attributes.timestamp > 1749006880539")
print(f"Traces after specific timestamp: {safe_len(traces)}")

# Get current time in milliseconds
current_time_ms = int(time.time() * 1000)
five_minutes_ago = current_time_ms - (5 * 60 * 1000)

traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {five_minutes_ago}"
)
print(f"Recent traces (last 5 min): {safe_len(traces)}")

# Timestamp aliases (all equivalent to timestamp_ms)
print("\n### Timestamp aliases (all equivalent to timestamp_ms):")
aliases = ["timestamp", "timestampMs", "created"]
for alias in aliases:
    traces = mlflow.search_traces(
        filter_string=f"attributes.{alias} > {five_minutes_ago}"
    )
    print(f"attributes.{alias}: {safe_len(traces)} traces")

print("\n### Search by execution time (engineer's examples):")
# Engineer's example - 5 seconds in milliseconds
traces = mlflow.search_traces(filter_string="attributes.execution_time > 5000")
print(f"Traces > 5 seconds (engineer's example): {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="attributes.execution_time_ms > 50")
print(f"Slow traces (>50ms): {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="attributes.execution_time_ms < 100")
print(f"Fast traces (<100ms): {safe_len(traces)}")

# Execution time aliases (all equivalent to execution_time_ms)
print("\n### Execution time aliases (all equivalent to execution_time_ms):")
exec_aliases = ["executionTimeMs", "executionTime", "latency", "execution_time"]
for alias in exec_aliases:
    traces = mlflow.search_traces(filter_string=f"attributes.{alias} > 50")
    print(f"attributes.{alias}: {safe_len(traces)} traces")

# =============================================================================
# 2. TAG FILTERING - Custom and MLflow tags
# =============================================================================

print("\n\n## 2. TAG FILTERING")
print("-" * 40)

print("\n### Search by custom tags (engineer's examples):")
# Engineer's example
traces = mlflow.search_traces(filter_string="tags.mytag = 'myvalue'")
print(f"Engineer's example 'mytag': {safe_len(traces)}")

# Our custom tags
traces = mlflow.search_traces(filter_string="tags.customer_id = 'C001'")
print(f"Customer C001 traces: {safe_len(traces)}")

# Debug: Let's see what tags are actually available
all_traces = mlflow.search_traces()
if safe_len(all_traces) > 0:
    sample_tags = safe_get_first_trace(all_traces).tags
    print(f"DEBUG: Sample trace tags: {sample_tags}")
    print(
        f"DEBUG: Available tag keys: {list(sample_tags.keys()) if isinstance(sample_tags, dict) else 'Not a dict'}"
    )

traces = mlflow.search_traces(filter_string="tags.environment = 'production'")
print(f"Production traces: {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="tags.session_type = 'chat'")
print(f"Chat session traces: {safe_len(traces)}")

print("\n### Search by MLflow system tags (dotted names need backticks):")
traces = mlflow.search_traces(
    filter_string="tags.`mlflow.traceName` = 'process_chat_request'"
)
print(f"Chat processing traces: {safe_len(traces)}")

traces = mlflow.search_traces(
    filter_string="tags.`mlflow.traceName` = 'analyze_sentiment'"
)
print(f"Sentiment analysis traces: {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="tags.`mlflow.user` = '7422288164188866'")
print(f"Traces from specific MLflow user: {safe_len(traces)}")

# =============================================================================
# 3. REQUEST METADATA FILTERING - trace_metadata fields (engineer's focus)
# =============================================================================

print("\n\n## 3. REQUEST METADATA FILTERING")
print("-" * 40)

print("\n### Search by response content (engineer's key example):")
# Engineer's example - exact response matching
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.traceOutputs` = '<your exact response here>'"
)
print(f"Exact response match: {safe_len(traces)}")

# Search for traces with any response content
traces = mlflow.search_traces(filter_string="metadata.`mlflow.traceOutputs` != ''")
print(f"Traces with output data: {safe_len(traces)}")

print("\n### Search by request content:")
# Search for traces with any input data
traces = mlflow.search_traces(filter_string="metadata.`mlflow.traceInputs` != ''")
print(f"Traces with input data: {safe_len(traces)}")

print("\n### Search by user (engineer's field names):")
traces = mlflow.search_traces(filter_string="metadata.`mlflow.user` = 'eric.peter'")
print(f"Traces from eric.peter: {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="metadata.`mlflow.user` != ''")
print(f"Traces with user info: {safe_len(traces)}")

print("\n### Search by source information:")
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.source.name` = 'search_traces_comprehensive.py'"
)
print(f"Traces from this script: {safe_len(traces)}")

traces = mlflow.search_traces(filter_string="metadata.`mlflow.source.type` = 'LOCAL'")
print(f"Local source traces: {safe_len(traces)}")

print("\n### Search by git information:")
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.source.git.branch` = 'main'"
)
print(f"Main branch traces: {safe_len(traces)}")

# Search by trace size (as string since metadata values are strings)
traces = mlflow.search_traces(filter_string="metadata.`mlflow.trace.sizeBytes` != ''")
print(f"Traces with size info: {safe_len(traces)}")

# =============================================================================
# 4. COMPLEX FILTERING - Combining multiple conditions
# =============================================================================

print("\n\n## 4. COMPLEX FILTERING WITH AND")
print("-" * 40)

print("\n### Successful traces from specific user:")
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK' AND metadata.`mlflow.user` != ''"
)
print(f"Successful traces with user: {safe_len(traces)}")

print("\n### Fast production traces:")
traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms < 100 AND tags.environment = 'production'"
)
print(f"Fast production traces: {safe_len(traces)}")

print("\n### Recent chat processing traces:")
traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {five_minutes_ago} AND tags.`mlflow.traceName` = 'process_chat_request'"
)
print(f"Recent chat traces: {safe_len(traces)}")

print("\n### Traces within execution time range:")
traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms > 50 AND attributes.execution_time_ms < 200"
)
print(f"Medium duration traces (50-200ms): {safe_len(traces)}")

# =============================================================================
# 5. ORDERING - Sorting results
# =============================================================================

print("\n\n## 5. ORDERING RESULTS")
print("-" * 40)

print("\n### Most recent traces first:")
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'", order_by=["attributes.timestamp_ms DESC"]
)
print(f"Recent successful traces: {safe_len(traces)}")
if safe_get_first_trace(traces) is not None:
    # Convert timestamp to readable format
    timestamp_ms = safe_get_first_trace(traces).request_time
    from datetime import datetime

    readable_time = datetime.fromtimestamp(timestamp_ms / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(f"Most recent trace time: {readable_time} ({timestamp_ms}ms)")

print("\n### Fastest traces first:")
traces = mlflow.search_traces(order_by=["attributes.execution_time_ms ASC"])
print(f"Fastest traces: {safe_len(traces)}")

print("\n### Multiple ordering criteria:")
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC", "attributes.execution_time_ms ASC"],
)
print(f"Recent first, then fastest: {safe_len(traces)}")

# =============================================================================
# 6. PRACTICAL USE CASES
# =============================================================================

print("\n\n## 6. PRACTICAL USE CASES")
print("-" * 40)

print("\n### Monitor failed traces:")
failed_traces = mlflow.search_traces(filter_string="attributes.status = 'ERROR'")
print(f"❌ Found {safe_len(failed_traces)} failed traces")

print("\n### Find slow performance issues:")
slow_traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms > 1000",
    order_by=["attributes.execution_time_ms DESC"],
)
print(f"🐌 Found {safe_len(slow_traces)} slow traces (>1s)")

print("\n### Audit specific user activity:")
user_traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.user` != ''",
    order_by=["attributes.timestamp_ms DESC"],
)
print(f"👤 Found {safe_len(user_traces)} traces with user info")

print("\n### Find traces with specific content:")
# This would search for traces containing specific input/output content
content_traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.traceInputs` != ''"
)
print(f"📝 Found {safe_len(content_traces)} traces with recorded inputs")

print("\n### Monitor production environment:")
prod_traces = mlflow.search_traces(
    filter_string="tags.environment = 'production'",
    order_by=["attributes.timestamp_ms DESC"],
)
print(f"🏭 Found {safe_len(prod_traces)} production traces")

# =============================================================================
# 7. ADVANCED PATTERNS
# =============================================================================

print("\n\n## 7. ADVANCED SEARCH PATTERNS")
print("-" * 40)

print("\n### Time range searches:")
one_hour_ago = current_time_ms - (60 * 60 * 1000)
traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {one_hour_ago} AND attributes.timestamp_ms < {current_time_ms}"
)
print(f"Traces from last hour: {safe_len(traces)}")

print("\n### Performance profiling:")
# Find traces by performance characteristics
fast_traces = mlflow.search_traces(filter_string="attributes.execution_time_ms < 50")
medium_traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms >= 50 AND attributes.execution_time_ms < 200"
)
slow_traces = mlflow.search_traces(filter_string="attributes.execution_time_ms >= 200")

print(f"Performance distribution:")
print(f"  Fast (<50ms): {safe_len(fast_traces)} traces")
print(f"  Medium (50-200ms): {safe_len(medium_traces)} traces")
print(f"  Slow (>=200ms): {safe_len(slow_traces)} traces")

print("\n### Search by trace name patterns:")
chat_traces = mlflow.search_traces(
    filter_string="tags.`mlflow.traceName` = 'process_chat_request'"
)
analysis_traces = mlflow.search_traces(
    filter_string="tags.`mlflow.traceName` = 'analyze_sentiment'"
)

print(f"Function usage:")
print(f"  Chat processing: {safe_len(chat_traces)} traces")
print(f"  Sentiment analysis: {safe_len(analysis_traces)} traces")

print("\n" + "=" * 80)
print("SEARCH EXAMPLES COMPLETED")
print("=" * 80)

print(
    """
## Key Takeaways (Based on Engineer's Specifications)

1. **Limited searchable fields**: Only 6 field types are searchable per the backend implementation
2. **Metadata focus**: Request metadata (including response content) is a primary search target
3. **Operator restrictions**: Only =, != for strings; =, <, <=, >, >= for numeric fields
4. **AND only**: Complex logical operations not supported
5. **Case sensitive**: All field names and values are case sensitive
6. **Backticks required**: For dotted field names like `mlflow.user`, `mlflow.traceName`
7. **Timestamp format**: Milliseconds since Unix epoch (engineer's example: 1749006880539)
8. **Execution time**: Measured in milliseconds (engineer's example: > 5000)

## Searchable Field Types (Complete List)

1. **Request Metadata**: `metadata.*` (=, != only) - ANY field in trace_metadata
2. **Tags**: `tags.*` (=, != only) - Custom and MLflow system tags  
3. **Trace Name**: `attributes.name` (=, != only) - Legacy, rarely used
4. **Trace Status**: `attributes.status` (=, != only) - OK, ERROR, etc.
5. **Creation Time**: `attributes.timestamp_ms` (=, <, <=, >, >=) + aliases
6. **Execution Time**: `attributes.execution_time_ms` (=, <, <=, >, >=) + aliases

## Engineer's Original Examples

✅ `metadata.\`mlflow.traceOutputs\` = '<your exact response here>'`
✅ `tags.mytag = 'myvalue'`  
✅ `attributes.name = 'foo'`
✅ `attributes.status = 'ERROR'`
✅ `attributes.timestamp > 1749006880539`
✅ `attributes.execution_time > 5000`

## Common Pitfalls to Avoid

❌ `traces.status = 'OK'` → ✅ `attributes.status = 'OK'`
❌ `mlflow.user = 'alice'` → ✅ `metadata.\`mlflow.user\` = 'alice'`  
❌ `status == 'OK'` → ✅ `attributes.status = 'OK'`
❌ `timestamp > '2024-01-01'` → ✅ `attributes.timestamp > 1704067200000`
❌ `execution_time_ms > 1000` → ✅ `attributes.execution_time_ms > 1000`
"""
)

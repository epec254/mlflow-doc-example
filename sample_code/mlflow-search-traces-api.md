# MLflow Search Traces API Documentation

## Overview

The `mlflow.search_traces()` API allows you to search through traces based on their metadata, tags, and core attributes. The search operates over the trace JSON structure and supports filtering and ordering with specific syntax requirements.

**Key Features:**
- Filter traces by status, timestamps, execution time, custom tags, and metadata
- Support for complex queries with AND logic
- Ordering and sorting capabilities
- Access to both system-generated and custom trace data

## Table of Contents

- [Trace Structure](#trace-structure)
- [Searchable Fields](#searchable-fields)
- [Supported Operators](#supported-operators)
- [Search Syntax Rules](#search-syntax-rules)
- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Practical Use Cases](#practical-use-cases)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

## Trace Structure

Each trace contains the following searchable sections based on the engineer's specification:

```json
{
    "trace": {
        "trace_info": {
            "trace_id": "tr-7373cc836269db16866b684d80932a67",
            "client_request_id": "tr-7373cc836269db16866b684d80932a67",
            "trace_location": {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {
                    "experiment_id": "1455626485744285"
                }
            },
            "request": "{\"messages\": [...], \"customer_id\": \"C001\", \"session_id\": \"...\"}",
            "response": "",
            "request_preview": "{\"messages\": [...], \"customer_id\": \"C001\", \"session_id\": \"...\"}",
            "response_preview": "",
            "request_time": "2025-06-04T00:17:06.901Z",  // -> attributes.timestamp_ms
            "execution_duration": "1.840s",              // -> attributes.execution_time_ms  
            "state": "ERROR",                            // -> attributes.status
            "trace_metadata": {  // -> metadata.*
                "mlflow.trace.sizeBytes": "12155",
                "mlflow.source.git.repoURL": "https://github.com/epec254/mlflow-example.git",
                "mlflow.source.git.branch": "main",
                "mlflow.trace_schema.version": "3",
                "mlflow.source.type": "LOCAL",
                "mlflow.source.name": "runner.py",
                "mlflow.user": "eric.peter",
                "mlflow.traceOutputs": "",               // Response content - searchable!
                "mlflow.traceInputs": "{\"messages\": [...], \"customer_id\": \"C001\", \"session_id\": \"...\"}", // Request content - searchable!
                "mlflow.source.git.commit": "04f4e124e1309c74650b32957475e6ed468642d4"
            },
            "tags": {  // -> tags.*
                "mlflow.artifactLocation": "dbfs:/databricks/mlflow-tracking/...",
                "mlflow.traceName": "process_chat_request",
                "mlflow.user": "7422288164188866"
            }
        }
    }
}
```

## Searchable Fields

Based on the engineer's specification, you can **only** search on these field types:

### 1. **Request Metadata** (trace_metadata fields)
**Access:** `metadata.*` prefix  
**Operators:** `=`, `!=` only  
**Description:** Search ANY field within `trace_metadata` including response content

| Key Fields | Description | Example Search |
|------------|-------------|----------------|
| `metadata.mlflow.traceOutputs` | **Response/output content** | `metadata.\`mlflow.traceOutputs\` = 'exact response here'` |
| `metadata.mlflow.traceInputs` | Request/input content | `metadata.\`mlflow.traceInputs\` != ''` |
| `metadata.mlflow.user` | User who created trace | `metadata.\`mlflow.user\` = 'eric.peter'` |
| `metadata.mlflow.source.name` | Source file | `metadata.\`mlflow.source.name\` = 'runner.py'` |
| `metadata.mlflow.source.git.commit` | Git commit | `metadata.\`mlflow.source.git.commit\` = '04f4e124e1...'` |
| `metadata.mlflow.trace.sizeBytes` | Trace size | `metadata.\`mlflow.trace.sizeBytes\` = '12155'` |

### 2. **Tags** (Custom and MLflow tags)
**Access:** `tags.*` prefix  
**Operators:** `=`, `!=` only  
**Description:** Search any tag set on the trace

| Tag Type | Examples |
|----------|----------|
| **Custom Tags** | `tags.customer_id = 'C001'`, `tags.environment = 'production'` |
| **MLflow System Tags** | `tags.\`mlflow.traceName\` = 'process_chat_request'`, `tags.\`mlflow.user\` = '7422288164188866'` |

### 3. **Trace Name** (Legacy - rarely used)
**Access:** `attributes.name`  
**Operators:** `=`, `!=` only  
**Description:** Named traces (most traces don't have names)

| Field | Example |
|-------|---------|
| `attributes.name` | `attributes.name = 'my_custom_trace_name'` |

### 4. **Trace Status** 
**Access:** `attributes.status`  
**Operators:** `=`, `!=` only  
**Description:** Trace execution state

| Field | Examples |
|-------|----------|
| `attributes.status` | `attributes.status = 'ERROR'`, `attributes.status = 'OK'` |

### 5. **Trace Creation Time**
**Access:** `attributes.timestamp_ms` (and aliases)  
**Operators:** `=`, `<`, `<=`, `>`, `>=`  
**Description:** When the trace was created (milliseconds since Unix epoch)

| Field | Aliases | Example |
|-------|---------|---------|
| `attributes.timestamp_ms` | `timestamp`, `timestampMs`, `created` | `attributes.timestamp > 1749006880539` |

### 6. **Trace Execution Time**
**Access:** `attributes.execution_time_ms` (and aliases)  
**Operators:** `=`, `<`, `<=`, `>`, `>=`  
**Description:** How long the trace took to execute (milliseconds)

| Field | Aliases | Example |
|-------|---------|---------|
| `attributes.execution_time_ms` | `executionTimeMs`, `executionTime`, `latency`, `execution_time` | `attributes.execution_time > 5000` |

## Supported Operators Summary

| Field Type | Supported Operators | Notes |
|------------|-------------------|-------|
| **Request Metadata** (`metadata.*`) | `=`, `!=` | String equality only |
| **Tags** (`tags.*`) | `=`, `!=` | String equality only |
| **Trace Name** (`attributes.name`) | `=`, `!=` | String equality only |
| **Trace Status** (`attributes.status`) | `=`, `!=` | String equality only |
| **Timestamp** (`attributes.timestamp_ms`) | `=`, `<`, `<=`, `>`, `>=` | Numeric comparisons |
| **Execution Time** (`attributes.execution_time_ms`) | `=`, `<`, `<=`, `>`, `>=` | Numeric comparisons |

### Logical Operators
- `AND` ✅ (supported)
- `OR` ❌ (NOT supported)

## Search Syntax Rules

1. **Table prefixes required**: Always use `attributes.`, `tags.`, or `metadata.`
2. **Quoted field names**: Use backticks for fields with special characters:
   ```
   metadata.`mlflow.user`
   tags.`mlflow.traceName`
   ```
3. **Single expressions or AND only**: Complex nested logic not supported
4. **Case sensitive**: All field names and values are case sensitive
5. **No fuzzy matching**: Exact string matches only

## Basic Examples

### Search by Status
```python
# Find failed traces (engineer's example)
traces = mlflow.search_traces(filter_string="attributes.status = 'ERROR'")

# Find successful traces  
traces = mlflow.search_traces(filter_string="attributes.status = 'OK'")
```

### Search by Timestamp
```python
# Engineer's example - timestamp in milliseconds since Unix epoch
traces = mlflow.search_traces(filter_string="attributes.timestamp > 1749006880539")

# Recent traces (last 5 minutes) 
five_minutes_ago = int(time.time() * 1000) - (5 * 60 * 1000)
traces = mlflow.search_traces(f"attributes.timestamp_ms > {five_minutes_ago}")

# Using aliases (all equivalent to timestamp_ms)
traces = mlflow.search_traces(f"attributes.timestamp > {five_minutes_ago}")
traces = mlflow.search_traces(f"attributes.timestampMs > {five_minutes_ago}")
traces = mlflow.search_traces(f"attributes.created > {five_minutes_ago}")
```

### Search by Execution Time
```python
# Engineer's example - execution time in milliseconds
traces = mlflow.search_traces("attributes.execution_time > 5000")

# Slow traces (>1 second using primary field name)
traces = mlflow.search_traces("attributes.execution_time_ms > 1000")

# Using aliases (all equivalent to execution_time_ms)
traces = mlflow.search_traces("attributes.executionTimeMs > 1000")
traces = mlflow.search_traces("attributes.latency > 1000")
traces = mlflow.search_traces("attributes.execution_time > 1000")
```

### Search by Trace Name (Legacy)
```python
# Engineer's example - trace name (rarely used)
traces = mlflow.search_traces("attributes.name = 'foo'")

# Most traces don't have names, so this often returns empty results
traces = mlflow.search_traces("attributes.name = 'my_custom_trace_name'")
```

### Search by Tags
```python
# Engineer's example - custom tag
traces = mlflow.search_traces("tags.mytag = 'myvalue'")

# Custom tags (set via mlflow.update_current_trace)
traces = mlflow.search_traces("tags.environment = 'production'")
traces = mlflow.search_traces("tags.customer_id = 'C001'")

# MLflow system tags (note the backticks for dotted names)
traces = mlflow.search_traces("tags.`mlflow.traceName` = 'process_chat_request'")
traces = mlflow.search_traces("tags.`mlflow.user` = '7422288164188866'")
```

### Search by Request Metadata
```python
# Engineer's example - search response content (exact match)
traces = mlflow.search_traces("metadata.`mlflow.traceOutputs` = '<your exact response here>'")

# Search by user
traces = mlflow.search_traces("metadata.`mlflow.user` = 'eric.peter'")

# Search by source file  
traces = mlflow.search_traces("metadata.`mlflow.source.name` = 'runner.py'")

# Search by git commit
traces = mlflow.search_traces("metadata.`mlflow.source.git.commit` = '04f4e124e1309c74650b32957475e6ed468642d4'")

# Search traces with input data (non-empty)
traces = mlflow.search_traces("metadata.`mlflow.traceInputs` != ''")

# Search by trace size
traces = mlflow.search_traces("metadata.`mlflow.trace.sizeBytes` = '12155'")

# Search by git branch
traces = mlflow.search_traces("metadata.`mlflow.source.git.branch` = 'main'")
```

## Advanced Examples

### Complex Filtering with AND
```python
# Successful traces from specific user
traces = mlflow.search_traces(
    "attributes.status = 'OK' AND metadata.`mlflow.user` = 'alice@company.com'"
)

# Fast production traces
traces = mlflow.search_traces(
    "attributes.execution_time_ms < 100 AND tags.environment = 'production'"
)

# Recent chat processing traces
traces = mlflow.search_traces(
    f"attributes.timestamp_ms > {recent_time} AND tags.`mlflow.traceName` = 'process_chat'"
)

# Performance range filtering
traces = mlflow.search_traces(
    "attributes.execution_time_ms > 100 AND attributes.execution_time_ms < 1000"
)
```

### Ordering Results
```python
# Most recent traces first
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"]
)

# Fastest traces first
traces = mlflow.search_traces(
    order_by=["attributes.execution_time_ms ASC"]
)

# Multiple ordering criteria
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC", "attributes.execution_time_ms ASC"]
)
```

## Practical Use Cases

### 1. Error Monitoring
```python
# Find all failed traces
failed_traces = mlflow.search_traces("attributes.status = 'ERROR'")

# Recent failures
recent_failures = mlflow.search_traces(
    f"attributes.status = 'ERROR' AND attributes.timestamp_ms > {one_hour_ago}"
)
```

### 2. Performance Analysis
```python
# Find slow traces (>5 seconds)
slow_traces = mlflow.search_traces(
    filter_string="attributes.execution_time_ms > 5000",
    order_by=["attributes.execution_time_ms DESC"]
)

# Performance distribution
fast_traces = mlflow.search_traces("attributes.execution_time_ms < 100")
medium_traces = mlflow.search_traces(
    "attributes.execution_time_ms >= 100 AND attributes.execution_time_ms < 1000"
)
slow_traces = mlflow.search_traces("attributes.execution_time_ms >= 1000")
```

### 3. User Activity Auditing
```python
# Traces from specific user
user_traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.user` = 'alice@company.com'",
    order_by=["attributes.timestamp_ms DESC"]
)

# Production environment activity
prod_traces = mlflow.search_traces(
    filter_string="tags.environment = 'production'",
    order_by=["attributes.timestamp_ms DESC"]
)
```

### 4. Function Usage Analysis
```python
# Specific function usage
function_traces = mlflow.search_traces(
    "tags.`mlflow.traceName` = 'process_payment'"
)

# Compare function performance
chat_traces = mlflow.search_traces("tags.`mlflow.traceName` = 'process_chat'")
analysis_traces = mlflow.search_traces("tags.`mlflow.traceName` = 'analyze_sentiment'")
```

### 5. Time Range Analysis
```python
# Last hour activity
one_hour_ago = int(time.time() * 1000) - (60 * 60 * 1000)
current_time = int(time.time() * 1000)

traces = mlflow.search_traces(
    f"attributes.timestamp_ms > {one_hour_ago} AND attributes.timestamp_ms < {current_time}"
)
```

## Setting Custom Tags

To make traces searchable by custom tags, use `mlflow.update_current_trace()` within your traced functions:

```python
@mlflow.trace
def process_request(customer_id, request_type):
    # Set custom tags for searchability
    mlflow.update_current_trace(tags={
        "customer_id": customer_id,
        "request_type": request_type,
        "environment": "production",
        "version": "v2.1.0"
    })
    
    # Your function logic here
    return result
```

## Return Value Handling

The `search_traces()` function can return either a pandas DataFrame or a list, depending on the MLflow version and configuration. Handle both cases:

```python
def safe_len(traces):
    """Get length regardless of return type"""
    return len(traces) if traces is not None else 0

def safe_get_first_trace(traces):
    """Safely get first trace from DataFrame or list"""
    if traces is None or len(traces) == 0:
        return None
    if hasattr(traces, 'iloc'):
        return traces.iloc[0]  # DataFrame
    elif isinstance(traces, list):
        return traces[0]       # List
    else:
        return None

# Usage
traces = mlflow.search_traces("attributes.status = 'OK'")
print(f"Found {safe_len(traces)} traces")

if safe_get_first_trace(traces) is not None:
    first_trace = safe_get_first_trace(traces)
    print(f"Most recent: {first_trace.request_time}")
```

## Common Pitfalls

### ❌ Incorrect vs ✅ Correct

| ❌ Wrong | ✅ Correct | Issue |
|----------|------------|-------|
| `traces.status = 'OK'` | `attributes.status = 'OK'` | Missing prefix |
| `status == 'OK'` | `attributes.status = 'OK'` | Wrong operator |
| `mlflow.user = 'alice'` | `metadata.\`mlflow.user\` = 'alice'` | Missing prefix & quotes |
| `timestamp > '2024-01-01'` | `attributes.timestamp > 1704067200000` | Wrong format (use engineer's pattern) |
| `execution_time_ms > 1000` | `attributes.execution_time_ms > 1000` | Missing prefix |
| `tags.environment = "prod"` | `tags.environment = 'prod'` | Wrong quote type |
| `status = 'OK' OR status = 'CANCELLED'` | Use separate queries | OR not supported |

### Common Issues

1. **Missing table prefixes**: Always use `attributes.`, `tags.`, or `metadata.`

2. **Special characters in field names**: Use backticks for dotted field names:
   ```python
   # ✅ Correct
   metadata.`mlflow.user`
   tags.`mlflow.traceName`
   ```

3. **Wrong timestamp format**: Use milliseconds since Unix epoch:
   ```python
   # ❌ Wrong
   "attributes.timestamp > '2024-01-01'"
   
   # ✅ Correct (engineer's pattern)
   timestamp_ms = int(datetime(2024, 1, 1).timestamp() * 1000)
   f"attributes.timestamp > {timestamp_ms}"
   # Example: attributes.timestamp > 1749006880539
   ```

4. **Case sensitivity**: Field names and values are case sensitive:
   ```python
   # These are different:
   "tags.Environment = 'Production'"  # Won't match
   "tags.environment = 'production'"  # Will match
   ```

5. **Limited search scope**: Only 6 field types are searchable according to the backend implementation

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

## Best Practices

### 1. Use Descriptive Custom Tags
```python
@mlflow.trace  
def process_order(order_id, customer_tier, region):
    mlflow.update_current_trace(tags={
        "order_id": order_id,
        "customer_tier": customer_tier,  # "premium", "standard"
        "region": region,                # "us-east", "eu-west"
        "function_type": "order_processing",
        "version": "v2.1.0"
    })
```

### 2. Standardize Tag Naming
- Use consistent naming conventions: `snake_case` or `kebab-case`
- Include version information for tracking changes
- Use categorical values for better filtering

### 3. Performance Considerations
- Be specific with filters to reduce result sets
- Use timestamp ranges for time-based queries
- Consider indexing frequently searched tags

### 4. Error Handling
```python
def search_traces_safely(filter_string, order_by=None):
    try:
        traces = mlflow.search_traces(
            filter_string=filter_string,
            order_by=order_by
        )
        return traces
    except Exception as e:
        print(f"Search failed: {e}")
        return []
```

### 5. Building Dynamic Queries
```python
def build_filter_query(status=None, user=None, min_duration=None, tags=None):
    conditions = []
    
    if status:
        conditions.append(f"attributes.status = '{status}'")
    
    if user:
        conditions.append(f"metadata.`mlflow.user` = '{user}'")
        
    if min_duration:
        conditions.append(f"attributes.execution_time_ms > {min_duration}")
    
    if tags:
        for key, value in tags.items():
            conditions.append(f"tags.{key} = '{value}'")
    
    return " AND ".join(conditions) if conditions else ""

# Usage
filter_query = build_filter_query(
    status="OK",
    user="alice@company.com", 
    min_duration=1000,
    tags={"environment": "production"}
)
traces = mlflow.search_traces(filter_query)
```

## API Reference

### Function Signature
```python
mlflow.search_traces(
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None,
    max_results: Optional[int] = None
) -> Union[pandas.DataFrame, List]
```

### Parameters
- **filter_string**: SQL-like filter expression
- **order_by**: List of column names with optional ASC/DESC
- **max_results**: Maximum number of traces to return

### Returns
- pandas DataFrame or List of trace objects (version dependent)

---

This documentation covers the complete MLflow Search Traces API functionality. For more examples and advanced use cases, refer to the comprehensive Python examples in the repository. 
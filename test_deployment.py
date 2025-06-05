#!/usr/bin/env python3
"""
Test script to verify the migrated Databricks app is working correctly.
Run this after setting up the required environment variables and Databricks CLI authentication.
"""

import requests
import json
import sys
import time
from databricks import sdk
import os

# Backend URL configuration - must be set via environment variable
BACKEND_URL = os.environ.get("BACKEND_URL")
if not BACKEND_URL:
    raise ValueError("BACKEND_URL environment variable must be set")
# Initialize Databricks SDK for authentication
w = sdk.WorkspaceClient()


def make_authenticated_request(method, url, **kwargs):
    """Make an authenticated request to the Databricks app"""
    headers = w.config.authenticate()
    print(headers)
    if "headers" in kwargs:
        kwargs["headers"].update(headers)
    else:
        kwargs["headers"] = headers

    return requests.request(method, url, **kwargs)


def test_health_endpoints():
    """Test basic health check endpoints"""
    print("Testing health endpoints...")

    # Test /api/health
    response = make_authenticated_request("GET", f"{BASE_URL}/api/health")
    print(response.status_code)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print(
        f"✓ /api/health working - OpenAI client initialized: {data['openai_client_initialized']}"
    )

    # Test /api/env-check
    response = make_authenticated_request("GET", f"{BASE_URL}/api/env-check")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ /api/env-check working - All vars present: {data['all_vars_present']}")
    if not data["all_vars_present"]:
        print("  Missing environment variables:")
        for key, value in data["environment_variables"].items():
            if value is None:
                print(f"    - {key}")


def test_company_endpoints():
    """Test company data endpoints"""
    print("\nTesting company endpoints...")

    # Test /api/companies
    response = make_authenticated_request("GET", f"{BASE_URL}/api/companies")
    assert response.status_code == 200
    companies = response.json()
    print(f"✓ /api/companies working - Found {len(companies)} companies")

    if companies:
        # Test getting a specific company
        company_name = companies[0]["name"]
        response = make_authenticated_request(
            "GET", f"{BASE_URL}/api/customer/{company_name}"
        )
        assert response.status_code == 200
        customer = response.json()
        print(f"✓ /api/customer/{company_name} working")
        return customer
    return None


def test_email_generation(customer_data):
    """Test email generation endpoint"""
    if not customer_data:
        print("\nSkipping email generation test - no customer data available")
        return

    print("\nTesting email generation...")

    # Prepare request
    request_data = {"customer_info": customer_data}

    try:
        # Test regular email generation with timing
        start_time = time.time()
        response = make_authenticated_request(
            "POST", f"{BASE_URL}/api/generate-email/", json=request_data
        )
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds

        if response.status_code == 200:
            email = response.json()
            print("✓ /api/generate-email/ working")
            print(f"  Subject: {email['subject_line'][:50]}...")
            print(f"  Body preview: {email['body'][:100]}...")
            print(f"  Latency: {latency:.2f}ms")
            if "trace_id" in email:
                print(f"  Trace ID: {email['trace_id']}")
        else:
            print(
                f"✗ Email generation failed: {response.status_code} - {response.text}"
            )
    except Exception as e:
        print(f"✗ Email generation error: {str(e)}")


def test_streaming_endpoint(customer_data):
    """Test streaming email generation endpoint"""
    if not customer_data:
        print("\nSkipping streaming test - no customer data available")
        return

    print("\nTesting streaming email generation...")

    # Prepare request
    request_data = {"customer_info": customer_data}

    try:
        start_time = time.time()
        response = make_authenticated_request(
            "POST",
            f"{BASE_URL}/api/generate-email-stream/",
            json=request_data,
            stream=True,
        )

        if response.status_code == 200:
            print("✓ /api/generate-email-stream/ working")
            token_count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                        if data.get("type") == "token":
                            token_count += 1
                        elif data.get("type") == "done":
                            latency = (time.time() - start_time) * 1000
                            print(f"  Received {token_count} tokens")
                            print(f"  Streaming latency: {latency:.2f}ms")
                            if "trace_id" in data:
                                print(f"  Trace ID: {data['trace_id']}")
                            break
        else:
            print(f"✗ Streaming failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Streaming error: {str(e)}")


def main():
    """Run all tests"""
    print("Starting Databricks App migration tests with authentication...\n")

    try:
        # Check if server is running and authentication works
        start_time = time.time()
        response = make_authenticated_request(
            "GET", f"{BASE_URL}/api/hello", timeout=10
        )
        latency = (time.time() - start_time) * 1000
        print(
            f"Initial connection test: Status={response.status_code}, Latency={latency:.2f}ms\n"
        )

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("Please check that the Databricks app is running and accessible")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Authentication or connection error: {str(e)}")
        print("Please ensure your Databricks CLI is configured:")
        print("  databricks auth login --profile DEFAULT")
        sys.exit(1)

    test_health_endpoints()
    customer_data = test_company_endpoints()
    test_email_generation(customer_data)
    test_streaming_endpoint(customer_data)

    print("\n✅ Migration tests completed!")


if __name__ == "__main__":
    main()

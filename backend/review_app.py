import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import mlflow
from openai import OpenAI
import json

from mlflow.entities import Document
from typing import List, Dict


mlflow.openai.autolog()

# Connect to Databricks LLM via OpenAI using the same credentials as MLflow
client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url=f"{os.environ.get('DATABRICKS_HOST')}/serving-endpoints",
)


# This function should be the same as what is called by your production application.
# It will be called by `evaluate(...)`.
@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str) -> List[Document]:
    # For example purposes, returning a mock document.
    # In a real scenario, this would query a vector database or search engine.
    if "john doe" in query.lower():
        return [
            Document(
                id="doc1",
                page_content="John Doe mentioned issues with login on July 10th. Expressed interest in feature X.",
                metadata={"doc_uri": "crm://conversations/123"},
            ),
            Document(
                id="doc2",
                page_content="Follow-up call with John Doe on July 12th. Login issue resolved. Discussed pricing for feature X.",
                metadata={"doc_uri": "crm://conversations/124"},
            ),
        ]
    return [
        Document(
            id="generic1",
            page_content="This is a generic document for other queries.",
            metadata={"doc_uri": "http://domain.com/default"},
        )
    ]


@mlflow.trace(span_type="TOOL")
def get_product_usage(customer_name: str) -> Dict:
    """
    Retrieves (mock) product usage data for a given customer.
    """
    print(f"Simulating API call to fetch product usage for: {customer_name}")
    # Example data structure based on customer name
    if "john doe" in customer_name.lower():
        return {
            "customer_id": customer_name,
            "active_users_last_7_days": 5,
            "key_features_used": ["Dashboard", "Reporting"],
            "last_login_date": "2024-07-18",
            "subscription_tier": "Premium",
        }
    elif "acme corp" in customer_name.lower():
        return {
            "customer_id": customer_name,
            "active_users_last_7_days": 150,
            "key_features_used": [
                "Dashboard",
                "Reporting",
                "Integration API",
                "Advanced Analytics",
            ],
            "last_login_date": "2024-07-19",
            "subscription_tier": "Enterprise",
        }
    else:
        return {
            "customer_id": customer_name,
            "error": "Product usage data not found for this customer.",
        }


@mlflow.trace
def my_app(customer_name: str, topic: str):
    """
    Generates a one-paragraph summary for a customer by retrieving conversations,
    fetching product usage via a tool, and then summarizing the combined information.
    """
    # 1. Retrieve conversations
    retrieved_documents = retrieve_docs(query=customer_name)
    conversations_text = "\\n".join([doc.page_content for doc in retrieved_documents])
    if not conversations_text:
        conversations_text = "No conversation history found."

    # 2. Define the tool for the LLM
    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "get_product_usage",
                "description": "Get the product usage data for a specific customer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "The name of the customer, e.g., 'John Doe' or 'Acme Corp'.",
                        }
                    },
                    "required": ["customer_name"],
                },
            },
        }
    ]

    # 3. Initial messages for the LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert sales assistant. Your primary goal is to generate a concise, "
                "one-paragraph summary about a customer, combining their conversation history "
                "and their product usage data. To achieve this, you MUST first call the "
                "'get_product_usage' tool to obtain the customer's product usage. "
                "Do not attempt to summarize or answer without this data. "
                "If the tool provides an error or no data, mention that in your summary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please generate a one-paragraph summary for customer '{customer_name}' "
                f"regarding our interactions on the topic of '{topic}'.\\n\\n"
                f"Here is their recent conversation history:\\n{conversations_text}"
            ),
        },
    ]

    # 4. First call to LLM - expected to trigger tool call
    response = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=messages,
        tools=tools_definition,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    # messages.append(response_message)  # Add assistant's response -- OLD WAY

    # Convert response_message object to a dictionary to maintain consistency in the messages list
    assistant_response_dict = {"role": response_message.role}
    if response_message.content:
        assistant_response_dict["content"] = response_message.content

    if response_message.tool_calls:
        assistant_response_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,  # Should be "function"
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in response_message.tool_calls
        ]
        # If tool_calls are present, content should be None as per OpenAI spec.
        # Explicitly set content to None if it was populated and tool_calls are also present.
        assistant_response_dict["content"] = None

    messages.append(assistant_response_dict)

    # 5. Check for tool calls and execute them
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_product_usage":
                function_args = json.loads(tool_call.function.arguments)
                tool_customer_name = function_args.get("customer_name", customer_name)

                product_usage_data = get_product_usage(customer_name=tool_customer_name)
                product_usage_content = json.dumps(product_usage_data)

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": product_usage_content,
                    }
                )
            # else: handle other tools if any in the future

        # 6. Second call to LLM with tool response to get the final summary
        final_response = client.chat.completions.create(
            model="databricks-claude-3-7-sonnet",
            messages=messages,
        )
        summary = final_response.choices[0].message.content
    else:
        summary = (
            response_message.content
            if response_message.content
            else "The model did not call the product usage tool as expected. No summary generated."
        )

    return {"summary": summary}


if __name__ == "__main__":
    print("--- Sample Call 1: John Doe, Billing Inquiry ---")
    # Ensure DATABRICKS_TOKEN and DATABRICKS_HOST are set in your environment or .env file
    try:
        summary1 = my_app(customer_name="John Doe", topic="Billing Inquiry")
        print(f"Summary for John Doe (Billing Inquiry):\n{summary1.get('summary')}\n")
    except Exception as e:
        print(f"Error during sample call 1: {e}")

    print("--- Sample Call 2: Acme Corp, New Feature Request ---")
    try:
        summary2 = my_app(customer_name="Acme Corp", topic="New Feature Request")
        print(
            f"Summary for Acme Corp (New Feature Request):\n{summary2.get('summary')}\n"
        )
    except Exception as e:
        print(f"Error during sample call 2: {e}")

    print("--- Sample Call 3: NonExistent Customer, General Query ---")
    try:
        summary3 = my_app(customer_name="No Such Customer Inc", topic="General Query")
        print(
            f"Summary for No Such Customer Inc (General Query):\n{summary3.get('summary')}\n"
        )
    except Exception as e:
        print(f"Error during sample call 3: {e}")

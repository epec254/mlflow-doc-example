#!/usr/bin/env python3
"""
Simulate production traffic to the email generation application.

This script:
- Reads records from input_data.jsonl
- For 30% of records: adds conflicting user instructions, 50% get thumbs down
- For 70% of records: no user input, 20% get thumbs up
"""

import json
import random
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
from databricks import sdk
import os

# Backend URL configuration - must be set via environment variable
BACKEND_URL = os.environ.get("BACKEND_URL")
if not BACKEND_URL:
    raise ValueError("BACKEND_URL environment variable must be set")

# Initialize Databricks SDK for authentication
w = sdk.WorkspaceClient()

# Conflicting instruction templates (go against the prompt guidelines)
CONFLICTING_INSTRUCTIONS = [
    "Make this email VERY sales-focused and pushy. Include lots of marketing language and push for an immediate sale.",
    "Don't mention any of the recent meetings or support tickets. Focus only on selling new features.",
    "Make this extremely formal and corporate. Use lots of jargon and buzzwords.",
    "Be super casual and use emojis 😊. Make it sound like a text message to a friend!",
    "Ignore all the customer data and just write a generic template email that could work for anyone.",
    "Focus ONLY on the problems and issues. Don't mention anything positive about their usage.",
    "Make the email very long and detailed. Include everything possible, at least 10 paragraphs.",
    "Skip the personalization. Don't use their name or reference their specific situation.",
    "Be aggressive about the least used features. Tell them they're missing out and need to use everything NOW.",
    "Don't suggest any next steps or meetings. Just send information with no call to action.",
]

# Feedback comments for thumbs down (when instructions were ignored)
THUMBS_DOWN_COMMENTS = [
    "The email didn't follow my instructions at all. I asked for a specific tone and it was ignored.",
    "I specifically asked to focus on sales but the email was too soft.",
    "My instructions about formality were completely ignored.",
    "The personalization request was not followed - this is still too generic.",
    "I asked for specific content focus but the AI did its own thing.",
    "The email tone is completely different from what I requested.",
    "None of my custom instructions were incorporated into the email.",
    "The AI seems to have ignored my input and followed its own template.",
    "My specific requests about content and style were not reflected in the output.",
    "This doesn't match what I asked for in my instructions.",
]


class TrafficSimulator:
    def __init__(self, backend_url: str = BACKEND_URL):
        self.backend_url = backend_url
        self.auth_headers = self._get_auth_headers()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "with_instructions": 0,
            "without_instructions": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "no_feedback": 0,
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from Databricks SDK"""
        return w.config.authenticate()

    async def parse_streaming_response(self, response) -> Tuple[Optional[str], str]:
        """Parse SSE streaming response to extract trace_id and collect content."""
        trace_id = None
        content = ""

        async for line in response.content:
            line = line.decode("utf-8").strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "token":
                        content += data.get("content", "")
                    elif data.get("type") == "done":
                        # Only set trace_id if we don't already have one and this one is not None
                        received_trace_id = data.get("trace_id")
                        if trace_id is None and received_trace_id is not None:
                            trace_id = received_trace_id
                    elif data.get("type") == "error":
                        # Only log non-None errors
                        error_msg = data.get("error")
                        if error_msg and error_msg != "INTERNAL_ERROR: None":
                            print(f"    ❌ Streaming error: {error_msg}")
                except json.JSONDecodeError:
                    continue

        return trace_id, content

    async def generate_email(
        self, session: aiohttp.ClientSession, customer_data: Dict, user_input: str = ""
    ) -> Optional[str]:
        """Call the email generation endpoint and return trace_id."""
        # Prepare request data
        request_data = {"customer_info": {**customer_data, "user_input": user_input}}

        # Combine auth headers with content-type
        headers = {**self.auth_headers, "Content-Type": "application/json"}

        try:
            async with session.post(
                f"{self.backend_url}/api/generate-email-stream/",
                json=request_data,
                headers=headers,
            ) as response:
                if response.status == 200:
                    trace_id, content = await self.parse_streaming_response(response)
                    if trace_id:
                        self.stats["successful_requests"] += 1
                        return trace_id
                    else:
                        print(f"  ⚠️  No trace_id received from response")
                        self.stats["failed_requests"] += 1
                        return None
                else:
                    response_text = await response.text()
                    print(
                        f"  ❌ Error generating email: {response.status} - {response_text}"
                    )
                    self.stats["failed_requests"] += 1
                    return None
        except Exception as e:
            print(f"  ❌ Exception during email generation: {e}")
            self.stats["failed_requests"] += 1
            return None

    async def submit_feedback(
        self,
        session: aiohttp.ClientSession,
        trace_id: str,
        rating: str,
        comment: str,
        sales_rep_name: str,
    ):
        """Submit feedback for a generated email."""
        feedback_data = {
            "trace_id": trace_id,
            "rating": rating,
            "comment": comment,
            "sales_rep_name": sales_rep_name,
        }

        # Combine auth headers with content-type
        headers = {**self.auth_headers, "Content-Type": "application/json"}

        try:
            async with session.post(
                f"{self.backend_url}/api/feedback",
                json=feedback_data,
                headers=headers,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        if rating == "up":
                            self.stats["thumbs_up"] += 1
                        else:
                            self.stats["thumbs_down"] += 1
                        return True
                    else:
                        print(
                            f"    ⚠️  Feedback submission failed: {result.get('message')}"
                        )
                        return False
                else:
                    response_text = await response.text()
                    print(
                        f"    ❌ Error submitting feedback: {response.status} - {response_text}"
                    )
                    return False
        except Exception as e:
            print(f"    ❌ Exception during feedback submission: {e}")
            return False

    async def process_record(
        self, session: aiohttp.ClientSession, record: Dict, index: int, total: int
    ):
        """Process a single customer record."""
        self.stats["total_requests"] += 1

        # Determine if this record should have user instructions (30%)
        has_instructions = random.random() < 0.3
        user_input = ""

        if has_instructions:
            user_input = random.choice(CONFLICTING_INSTRUCTIONS)
            self.stats["with_instructions"] += 1
            print(
                f"[{index+1}/{total}] Processing {record['account']['name']} WITH instructions"
            )
        else:
            self.stats["without_instructions"] += 1
            print(
                f"[{index+1}/{total}] Processing {record['account']['name']} without instructions"
            )

        # Generate email
        trace_id = await self.generate_email(session, record, user_input)

        if not trace_id:
            print(f"  ❌ Failed to generate email")
            return

        print(f"  ✅ Email generated (trace_id: {trace_id})")

        # Determine feedback
        sales_rep_name = record.get("sales_rep", {}).get("name", "Unknown")

        if has_instructions:
            # 50% chance of thumbs down for records with conflicting instructions
            if random.random() < 0.5:
                comment = random.choice(THUMBS_DOWN_COMMENTS)
                # Add a small delay to ensure trace is available
                await asyncio.sleep(1)
                success = await self.submit_feedback(
                    session, trace_id, "down", comment, sales_rep_name
                )
                if success:
                    print(f"  👎 Thumbs down feedback submitted")
                else:
                    print(f"  ❌ Failed to submit thumbs down feedback")
                    self.stats["no_feedback"] += 1
            else:
                self.stats["no_feedback"] += 1
                print(f"  ⏭️  No feedback provided")
        else:
            # 20% chance of thumbs up for records without instructions
            if random.random() < 0.2:
                # Add a small delay to ensure trace is available
                await asyncio.sleep(1)
                success = await self.submit_feedback(
                    session, trace_id, "up", "", sales_rep_name
                )
                if success:
                    print(f"  👍 Thumbs up feedback submitted")
                else:
                    print(f"  ❌ Failed to submit thumbs up feedback")
                    self.stats["no_feedback"] += 1
            else:
                self.stats["no_feedback"] += 1
                print(f"  ⏭️  No feedback provided")

        # Small delay to avoid overwhelming the server
        await asyncio.sleep(0.5)

    async def run_simulation(
        self, limit: Optional[int] = None, delay_between_batches: int = 5
    ):
        """Run the traffic simulation."""
        print(f"Starting traffic simulation with Databricks authentication...")
        print(f"Backend URL: {self.backend_url}")

        # Test authentication first
        try:
            print("Testing authentication...")
            async with aiohttp.ClientSession() as test_session:
                async with test_session.get(
                    f"{self.backend_url}/api/health", headers=self.auth_headers
                ) as response:
                    if response.status == 200:
                        print("✅ Authentication successful")
                    else:
                        print(f"❌ Authentication failed: {response.status}")
                        return
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            print("Please ensure your Databricks CLI is configured:")
            print("  databricks auth login --profile DEFAULT")
            return

        print(f"Reading customer data from input_data.jsonl...")

        # Load customer records
        records = []
        with open("input_data.jsonl", "r") as f:
            for line in f:
                records.append(json.loads(line.strip()))

        total_records = len(records)
        if limit:
            records = records[:limit]
            print(f"Limiting to {limit} records out of {total_records} total")
        else:
            print(f"Processing all {total_records} records")

        print("\nSimulation Configuration:")
        print("- 30% of records will have conflicting user instructions")
        print("- Of those with instructions: 50% will get thumbs down feedback")
        print("- 70% of records will have no user instructions")
        print("- Of those without instructions: 20% will get thumbs up feedback")
        print("\n" + "=" * 60 + "\n")

        start_time = time.time()

        # Process records in batches to avoid connection issues
        batch_size = 10
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                tasks = []

                for j, record in enumerate(batch):
                    task = self.process_record(session, record, i + j, len(records))
                    tasks.append(task)

                await asyncio.gather(*tasks)

                # Delay between batches
                if i + batch_size < len(records):
                    print(
                        f"\n⏸️  Pausing {delay_between_batches} seconds before next batch...\n"
                    )
                    await asyncio.sleep(delay_between_batches)

        end_time = time.time()
        duration = end_time - start_time

        # Print statistics
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        print(f"\nRequest Types:")
        print(
            f"  With instructions: {self.stats['with_instructions']} ({self.stats['with_instructions']/self.stats['total_requests']*100:.1f}%)"
        )
        print(
            f"  Without instructions: {self.stats['without_instructions']} ({self.stats['without_instructions']/self.stats['total_requests']*100:.1f}%)"
        )
        print(f"\nFeedback:")
        print(f"  Thumbs up: {self.stats['thumbs_up']}")
        print(f"  Thumbs down: {self.stats['thumbs_down']}")
        print(f"  No feedback: {self.stats['no_feedback']}")

        if self.stats["with_instructions"] > 0:
            thumbs_down_rate = (
                self.stats["thumbs_down"] / self.stats["with_instructions"] * 100
            )
            print(
                f"\nThumb down rate for requests with instructions: {thumbs_down_rate:.1f}%"
            )

        if self.stats["without_instructions"] > 0:
            thumbs_up_rate = (
                self.stats["thumbs_up"] / self.stats["without_instructions"] * 100
            )
            print(
                f"Thumbs up rate for requests without instructions: {thumbs_up_rate:.1f}%"
            )


async def main():
    parser = argparse.ArgumentParser(
        description="Simulate production traffic for email generation app"
    )
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    parser.add_argument(
        "--backend-url",
        default=BACKEND_URL,
        help=f"Backend URL (default: {BACKEND_URL})",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay between batches in seconds (default: 5)",
    )

    args = parser.parse_args()

    simulator = TrafficSimulator(backend_url=args.backend_url)
    await simulator.run_simulation(limit=args.limit, delay_between_batches=args.delay)


if __name__ == "__main__":
    asyncio.run(main())

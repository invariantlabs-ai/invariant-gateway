"""Test the chat completions proxy calls without tool calling."""

import os
import sys
import uuid

import pytest
from httpx import Client

# add tests folder (parent) to sys.path
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import *  # needed for pytest fixtures

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
async def test_chat_completion_without_streaming(context, explorer_api_url, proxy_url):
    """Test the chat completions proxy calls without tool calling."""
    dataset_name = "test-dataset-open-ai-" + str(uuid.uuid4())

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{proxy_url}/api/v1/proxy/{dataset_name}/openai",
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Verify the chat response
    assert "PARIS" in chat_response.choices[0].message.content.upper()

    # Fetch the trace ids for the dataset
    traces_response = await context.request.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces"
    )
    traces = await traces_response.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    # Fetch the trace
    trace_response = await context.request.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}"
    )
    trace = await trace_response.json()

    # Verify the trace messages
    assert len(trace["messages"]) == 2
    assert trace["messages"][0] == {
        "role": "user",
        "content": "What is the capital of France?",
    }
    assert trace["messages"][1]["role"] == "assistant"
    assert "PARIS" in trace["messages"][1]["content"].upper()

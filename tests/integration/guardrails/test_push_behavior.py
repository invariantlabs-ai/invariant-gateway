"""Test the Invariant-Push header behaviors with the OpenAI route."""

import os
import sys
import uuid
import time

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from httpx import Client
from openai import OpenAI

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_behavior",
    [(True, "push"), (True, "skip"), (False, "push"), (False, "skip")],
)
async def test_push_behavior(explorer_api_url, gateway_url, do_stream, push_behavior):
    """Test the Invariant-Push header behaviors."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-push-behavior-{uuid.uuid4()}"

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}",
                "Invariant-Push": push_behavior,
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai",
    )

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello!"}],
    }

    # Make the request
    chat_response = client.chat.completions.create(
        **request,
        stream=do_stream,
    )

    # If streaming, consume the stream
    if do_stream:
        response_content = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
    else:
        response_content = chat_response.choices[0].message.content

    assert response_content, "Response should not be empty"

    # Wait for potential trace saving
    time.sleep(2)

    # Check if trace was saved based on push_behavior
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()

    if push_behavior == "push":
        assert len(traces) == 1, "Trace should be saved when push_behavior is 'push'"
        
        # Verify trace contents
        trace_id = traces[0]["id"]
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}",
            timeout=5,
        )
        trace = trace_response.json()
        
        assert len(trace["messages"]) == 2, "Trace should contain both request and response"
        assert trace["messages"][0]["role"] == "user"
        assert trace["messages"][0]["content"] == "Say hello!"
        assert trace["messages"][1]["role"] == "assistant"
        assert trace["messages"][1]["content"] in response_content
    else:  # push_behavior == "skip"
        assert len(traces) == 0, "No trace should be saved when push_behavior is 'skip'"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
async def test_invalid_push_behavior(explorer_api_url, gateway_url):
    """Test invalid Invariant-Push header value."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-push-behavior-{uuid.uuid4()}"

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}",
                "Invariant-Push": "invalid_value",  # Invalid push behavior
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai",
    )

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello!"}],
    }

    # The request should still work, defaulting to "push" behavior
    chat_response = client.chat.completions.create(
        **request,
        stream=False,
    )

    assert chat_response.choices[0].message.content, "Response should not be empty"

    # Wait for trace saving
    time.sleep(2)

    # Check if trace was saved (should default to push behavior)
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 1, "Trace should be saved when push_behavior is invalid (defaults to push)" 
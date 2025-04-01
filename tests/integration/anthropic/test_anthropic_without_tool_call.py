"""Tests for the Anthropic API without tool call."""

import os
import sys
import time
import uuid

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
import pytest
import requests
from httpx import Client

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_response_without_tool_call(
    explorer_api_url, gateway_url, push_to_explorer
):
    """Test the Anthropic gateway without tool calling."""
    dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"
    invariant_api_key = os.environ.get("INVARIANT_API_KEY")

    client = anthropic.Anthropic(
        http_client=Client(
            headers={"Invariant-Authorization": f"Bearer {invariant_api_key}"},
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/anthropic",
    )

    cities = ["zurich", "new york", "london"]
    queries = [
        "Can you introduce Zurich, Switzerland within 200 words?",
        "Tell me the history of New York within 100 words?",
        "How's the weather in London next week?",
    ]

    # Process each query
    responses = []
    for query in queries:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
        )
        response_text = response.content[0].text
        responses.append(response_text)
        assert response_text is not None
        assert cities[queries.index(query)] in response_text.lower()

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == len(queries)

        for index, trace in enumerate(traces):
            trace_id = trace["id"]
            # Fetch the trace
            trace_response = requests.get(
                f"{explorer_api_url}/api/v1/trace/{trace_id}",
                timeout=5,
            )
            trace = trace_response.json()
            assert trace["messages"] == [
                {"role": "user", "content": queries[index]},
                {"role": "assistant", "content": responses[index]},
            ]


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_streaming_response_without_tool_call(
    explorer_api_url, gateway_url, push_to_explorer
):
    """Test the Anthropic gateway without tool calling."""
    dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"
    invariant_api_key = os.environ.get("INVARIANT_API_KEY")

    client = anthropic.Anthropic(
        http_client=Client(
            headers={"Invariant-Authorization": f"Bearer {invariant_api_key}"},
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/anthropic",
    )

    cities = ["zurich", "new york", "london"]
    queries = [
        "Can you introduce Zurich, Switzerland within 200 words?",
        "Tell me the history of New York within 100 words?",
        "How's the weather in London next week?",
    ]
    # Process each query
    responses = []
    for index, query in enumerate(queries):
        messages = [{"role": "user", "content": query}]
        response_text = ""

        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages,
        ) as response:
            for reply in response.text_stream:
                response_text += reply
            assert cities[index] in response_text.lower()
        responses.append(response_text)
        assert response_text is not None
        assert cities[queries.index(query)] in response_text.lower()

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == len(queries)

        for index, trace in enumerate(traces):
            trace_id = trace["id"]
            # Fetch the trace
            trace_response = requests.get(
                f"{explorer_api_url}/api/v1/trace/{trace_id}",
                timeout=5,
            )
            trace = trace_response.json()
            assert trace["messages"] == [
                {"role": "user", "content": queries[index]},
                {"role": "assistant", "content": responses[index]},
            ]

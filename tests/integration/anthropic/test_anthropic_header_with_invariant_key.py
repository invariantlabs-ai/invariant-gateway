"""Test the Anthropic gateway with Invariant key in the ANTHROPIC_API_KEY."""

import os
import sys
import time
import uuid
from unittest.mock import patch

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
async def test_gateway_with_invariant_key_in_anthropic_key_header(
    gateway_url, explorer_api_url
):
    """Test the Anthropic gateway with Invariant key in the Anthropic key"""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"
    invariant_key_suffix = f";invariant-auth={os.getenv('INVARIANT_API_KEY')}"
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": anthropic_api_key + invariant_key_suffix},
    ):
        client = anthropic.Anthropic(
            http_client=Client(),
            base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic",
        )
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Give me an introduction to Zurich, Switzerland within 200 words.",
                }
            ],
        )

        assert response is not None
        response_text = response.content[0].text
        assert "zurich" in response_text.lower()

        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)

        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == 1

        trace_id = traces[0]["id"]
        get_trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
        )
        trace = get_trace_response.json()
        assert trace["messages"] == [
            {
                "role": "user",
                "content": "Give me an introduction to Zurich, Switzerland within 200 words.",
            },
            {"role": "assistant", "content": response_text},
        ]

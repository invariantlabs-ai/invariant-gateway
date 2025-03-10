"""Test the Anthropic gateway with Invariant key in the ANTHROPIC_API_KEY."""

import datetime
import os
import sys
import time
from unittest.mock import patch

# Add tests folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
import pytest
from httpx import Client

from util import *  # Needed for pytest fixtures

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
async def test_gateway_with_invariant_key_in_anthropic_key_header(
    context, gateway_url, explorer_api_url
):
    """Test the Anthropic gateway with Invariant key in the Anthropic key"""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    dataset_name = "claude_header_test" + str(
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": anthropic_api_key
            + ";invariant-auth=<not needed for test>"
        },
    ):
        client = anthropic.Anthropic(
            http_client=Client(),
            base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic",
        )
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
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

        traces_response = await context.request.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces"
        )
        traces = await traces_response.json()
        assert len(traces) == 1

        trace_id = traces[0]["id"]
        get_trace_response = await context.request.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}"
        )
        trace = await get_trace_response.json()
        assert trace["messages"] == [
            {
                "role": "user",
                "content": "Give me an introduction to Zurich, Switzerland within 200 words.",
            },
            {"role": "assistant", "content": response_text},
        ]

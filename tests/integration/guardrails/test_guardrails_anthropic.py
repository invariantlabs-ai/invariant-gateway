"""Test the guardrails from file with the Anthropic route."""

import os
import sys
import uuid
import time

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from httpx import Client
from anthropic import Anthropic, APIStatusError, BadRequestError

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_message_content_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("GUARDRAILS_API_KEY"):
        pytest.fail("No GUARDRAILS_API_KEY set, failing")

    dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"

    client = Anthropic(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('GUARDRAILS_API_KEY')}"
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/anthropic",
    )

    request = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is the capital of Spain?"}],
    }

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.messages.create(
                **request,
                stream=False,
            )

        assert exc_info.value.status_code == 400
        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        assert "Madrid detected in the response" in str(exc_info.value)

    else:
        with pytest.raises(APIStatusError) as exc_info:
            chat_response = client.messages.create(
                **request,
                stream=True,
            )

            for _ in chat_response:
                pass
        assert (
            "[Invariant] The response did not pass the guardrails"
            in exc_info.value.message
        )
        assert "Madrid detected in the response" in str(exc_info.value.body)

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)

        # Fetch the trace ids for the dataset
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == 1
        trace_id = traces[0]["id"]

        # Fetch the trace
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}",
            timeout=5,
        )
        trace = trace_response.json()

        assert len(trace["messages"]) == 2
        assert trace["messages"][0] == {
            "role": "user",
            "content": "What is the capital of Spain?",
        }

        # Fetch annotations
        annotations_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
            timeout=5,
        )
        annotations = annotations_response.json()

        assert len(annotations) == 2
        assert (
            annotations[0]["content"] == "Madrid detected in the response"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )
        assert (
            annotations[1]["content"] == "Madrid detected in the response"
            and annotations[1]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_tool_call_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("GUARDRAILS_API_KEY"):
        pytest.fail("No GUARDRAILS_API_KEY set, failing")

    tools = [
        {
            "name": "get_capital",
            "description": "Get the capital of a country.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "country_name": {
                        "type": "string",
                        "description": "The name of the country we want the capital for.",
                    }
                },
                "required": ["country_name"],
            },
        }
    ]
    system_message = "Use the get_capital tool call to get the capital of a country. If the user input doesn't contain a country name, fail the request with a pretty message. If the get_capital tool call returns 'not_found' then fail the request with a pretty message. Do not return the capital if the get_capital tool call returns 'not_found'."
    request = {
        "messages": [
            {"role": "user", "content": "What is the capital of Germany?"},
        ],
        "tools": tools,
        "system": system_message,
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 150,
    }

    dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"

    client = Anthropic(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('GUARDRAILS_API_KEY')}"
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/anthropic",
    )

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.messages.create(**request, stream=False)

        assert exc_info.value.status_code == 400
        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        assert "get_capital is called with Germany as argument" in str(exc_info.value)

    else:
        with pytest.raises(APIStatusError) as exc_info:
            chat_response = client.messages.create(**request, stream=True)

            for _ in chat_response:
                pass
        assert (
            "[Invariant] The response did not pass the guardrails"
            in exc_info.value.message
        )
        assert "get_capital is called with Germany as argument" in str(
            exc_info.value.body
        )

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)

        # Fetch the trace ids for the dataset
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == 1
        trace_id = traces[0]["id"]

        # Fetch the trace
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}",
            timeout=5,
        )
        trace = trace_response.json()

        assert len(trace["messages"]) >= 3
        assert trace["messages"][0] == {"role": "system", "content": system_message}
        assert trace["messages"][1] == {
            "role": "user",
            "content": "What is the capital of Germany?",
        }

        # Fetch annotations
        annotations_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
            timeout=5,
        )
        annotations = annotations_response.json()

        assert len(annotations) == 1
        assert (
            annotations[0]["content"]
            == "get_capital is called with Germany as argument"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )

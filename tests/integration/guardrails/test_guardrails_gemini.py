"""Test the guardrails from file with the Gemini route."""

import os
import sys
import uuid
import time

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from httpx import Client
from google import genai

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_message_content_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "headers": {
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
        },
    )

    request = {
        "model": "gemini-2.0-flash",
        "contents": "What is the capital of Ireland?",
        "config": {
            "maxOutputTokens": 200,
        },
    }

    if not do_stream:
        with pytest.raises(genai.errors.ClientError) as exc_info:
            response = client.models.generate_content(
                **request,
            )
            assert "[Invariant] The response did not pass the guardrails" in str(
                exc_info
            )
            assert "Dublin detected in the response" in str(exc_info)

    else:
        response = client.models.generate_content_stream(**request)
        for chunk in response:
            assert "Dublin" not in str(chunk)

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
            "content": [{"type": "text", "text": "What is the capital of Ireland?"}],
        }

        # Fetch annotations
        annotations_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
            timeout=5,
        )
        annotations = annotations_response.json()

        assert len(annotations) == 1
        assert (
            annotations[0]["content"] == "Dublin detected in the response"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_tool_call_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    def get_capital(country_name: str) -> str:
        """Given a country name, return the capital of the country. (Mock API)

        Args:
            country_name: The name of the country we want the capital for.

        Returns:
            A string containing the capital of the country.
        """
        return "Something"

    config = genai.types.GenerateContentConfig(
        tools=[get_capital],
        system_instruction="This the system instruction. Use the function call to find the capital of a country.",
    )

    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "headers": {
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
        },
    )

    request = {
        "model": "gemini-2.0-flash",
        "contents": "What is the capital of Germany?",
        "config": config,
    }

    if not do_stream:
        with pytest.raises(genai.errors.ClientError) as exc_info:
            client.models.generate_content(
                **request,
            )

            assert exc_info.value.status_code == 400
            assert "[Invariant] The response did not pass the guardrails" in str(
                exc_info
            )
            assert "get_capital is called with Germany as argument" in str(exc_info)

    else:
        response = client.models.generate_content_stream(
            **request,
        )

        for chunk in response:
            assert "Madrid" not in str(chunk)

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
        assert trace["messages"][0] == {
            "role": "system",
            "content": config.system_instruction,
        }
        assert trace["messages"][1] == {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Germany?"}],
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

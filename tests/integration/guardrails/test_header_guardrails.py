"""Test the guardrails from header with the OpenAI route."""

import os
import sys
import uuid
import time

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from httpx import Client
from openai import OpenAI, BadRequestError, APIError

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_input_guardrail_in_header(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"

    policy = """
# For input guardrailing specifically
raise "Users must not mention the magic phrase 'Abracadabra'" if:
    (msg: Message)
    msg.role == "user"
    "Abracadabra" in msg.content
"""

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}",
                "Invariant-Guardrails": policy.encode("unicode-escape"),
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me more about Abracadabra."}],
    }

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=False,
            )

        assert exc_info.value.status_code == 400
        assert "[Invariant] The request did not pass the guardrails" in str(
            exc_info.value
        )
        assert "Users must not mention the magic phrase 'Abracadabra'" in str(
            exc_info.value
        )

    else:
        with pytest.raises(APIError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=True,
            )

            for _ in chat_response:
                pass
        assert (
            "[Invariant] The request did not pass the guardrails"
            in exc_info.value.message
        )
        assert "Users must not mention the magic phrase 'Abracadabra'" in str(
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

        # in case of input guardrailing, the pushed trace will not contain a response
        assert len(trace["messages"]) == 1
        assert trace["messages"][0] == {
            "role": "user",
            "content": "Tell me more about Abracadabra.",
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
            == "Users must not mention the magic phrase 'Abracadabra'"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_invalid_guardrail_in_header(gateway_url, do_stream, push_to_explorer):
    """Test the message content guardrail."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"

    policy = """
# For input guardrailing specifically
raise "Users must not mention the magic phrase 'Abracadabra'" if:
    (msg: Message)
    msg.role == "user"
    "Abracadabra" in msg.content
    illegal statement
"""

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}",
                "Invariant-Guardrails": policy.encode("unicode-escape"),
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me more about Abracadabra."}],
    }

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=False,
            )

        print(exc_info.value.message, flush=True)
        assert "Failed to create policy from policy source." in str(
            exc_info.value
        ), "guardrails check fails because of an invalid guardrailing rule"
        assert "illegal statement" in str(
            exc_info.value
        ), "error points to illegal statement in the rule definition"

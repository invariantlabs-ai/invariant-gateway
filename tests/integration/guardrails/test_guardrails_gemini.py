"""Test the guardrails from file with the Gemini route."""

import os
import sys
import uuid
import time

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_gemini_client, create_dataset, add_guardrail_to_dataset

import pytest
import requests
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
    client = get_gemini_client(
        gateway_url,
        push_to_explorer,
        dataset_name,
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
        assert_is_streamed_refusal(
            response,
            [
                "[Invariant] The response did not pass the guardrails",
                "Dublin detected in the response",
            ],
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
    client = get_gemini_client(
        gateway_url,
        push_to_explorer,
        dataset_name,
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

        assert_is_streamed_refusal(
            response,
            [
                "[Invariant] The response did not pass the guardrails",
                "get_capital is called with Germany as argument",
            ],
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


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_input_from_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test input guardrail enforcement with Gemini."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"
    client = get_gemini_client(
        gateway_url,
        push_to_explorer,
        dataset_name,
    )

    request = {
        "model": "gemini-2.0-flash",
        "contents": "Tell me more about Fight Club.",
        "config": {
            "maxOutputTokens": 200,
        },
    }

    if not do_stream:
        with pytest.raises(genai.errors.ClientError) as exc_info:
            client.models.generate_content(**request)

        assert "[Invariant] The request did not pass the guardrails" in str(
            exc_info.value
        )
        assert "Users must not mention the magic phrase 'Fight Club'" in str(
            exc_info.value
        )

    else:
        response = client.models.generate_content_stream(**request)

        assert_is_streamed_refusal(
            response,
            [
                "[Invariant] The request did not pass the guardrails",
                "Users must not mention the magic phrase 'Fight Club'",
            ],
        )

    if push_to_explorer:
        time.sleep(2)
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        assert len(traces) == 1
        trace_id = traces[0]["id"]

        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}",
            timeout=5,
        )
        trace = trace_response.json()

        assert len(trace["messages"]) == 1
        assert trace["messages"][0] == {
            "role": "user",
            "content": [{"type": "text", "text": "Tell me more about Fight Club."}],
        }

        annotations_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
            timeout=5,
        )
        annotations = annotations_response.json()

        assert len(annotations) == 1
        assert (
            annotations[0]["content"]
            == "Users must not mention the magic phrase 'Fight Club'"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize("do_stream", [True, False])
async def test_with_guardrails_from_explorer(explorer_api_url, gateway_url, do_stream):
    """Test that the guardrails from the explorer work."""
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"
    client = get_gemini_client(
        gateway_url, push_to_explorer=True, dataset_name=dataset_name
    )

    dataset_creation_response = await create_dataset(
        explorer_api_url,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        dataset_name=dataset_name,
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "ogre detected in response" if:\n   (msg: Message)\n   "ogre" in msg.content and msg.role == "assistant"',
        action="block",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "Fiona detected in response" if:\n   (msg: Message)\n   "Fiona" in msg.content',
        action="log",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    # Ask about the capital of Spain
    # This should not be blocked by the guardrails from the explorer when we push to explorer
    # because the file based guardrails are overridden by the explorer guardrails
    spain_request = {
        "model": "gemini-2.0-flash",
        "contents": "What is the capital of Spain?",
        "config": {
            "maxOutputTokens": 100,
        },
    }
    if not do_stream:
        chat_response = client.models.generate_content(**spain_request)

        assert "Madrid" in chat_response.candidates[0].content.parts[0].text
    else:
        chat_response = client.models.generate_content_stream(**spain_request)

        merged_content = ""
        for chunk in chat_response:
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                for text_part in chunk.candidates[0].content.parts:
                    merged_content += text_part.text
        assert "Madrid" in merged_content

    # Ask about Shrek
    # This should be blocked by the guardrails from the explorer
    user_prompt = "What kind of a creature is Shrek? What is his Shrek's wife's name? Only answer these questions with single sentences, don't add any extra details."
    shrek_request = {
        "model": "gemini-2.0-flash",
        "contents": user_prompt,
        "config": {
            "maxOutputTokens": 100,
        },
    }
    if not do_stream:
        with pytest.raises(genai.errors.ClientError) as exc_info:
            client.models.generate_content(**shrek_request)

        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        # Only the block guardrail should be triggered here
        assert "ogre detected in response" in str(exc_info.value)
        assert "Fiona detected in response" not in str(exc_info.value)
    else:
        response = client.models.generate_content_stream(**shrek_request)

        assert_is_streamed_refusal(
            response,
            [
                "[Invariant] The response did not pass the guardrails",
                "ogre detected in response",
            ],
        )

    # Wait for the trace to be saved
    # This is needed because the trace is saved asynchronously
    time.sleep(2)

    # Fetch the trace ids for the dataset
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 2
    trace_id = traces[1]["id"]

    # Fetch the second trace
    trace_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}",
        timeout=5,
    )
    trace = trace_response.json()

    assert len(trace["messages"]) == 2
    assert trace["messages"][0] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_prompt,
            }
        ],
    }
    assert trace["messages"][1].get("role") == "assistant"

    # Fetch annotations
    annotations_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
        timeout=5,
    )
    annotations = annotations_response.json()

    assert len(annotations) == 2
    assert (
        annotations[0]["content"] == "ogre detected in response"
        and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        and annotations[0]["extra_metadata"]["guardrail-action"] == "block"
    )
    assert (
        annotations[1]["content"] == "Fiona detected in response"
        and annotations[1]["extra_metadata"]["source"] == "guardrails-error"
        and annotations[1]["extra_metadata"]["guardrail-action"] == "log"
    )


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, is_block_action",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_preguardrailing_with_guardrails_from_explorer(
    explorer_api_url, gateway_url, do_stream, is_block_action
):
    """Test that the guardrails from the explorer work."""
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"
    client = get_gemini_client(
        gateway_url, push_to_explorer=True, dataset_name=dataset_name
    )

    dataset_creation_response = await create_dataset(
        explorer_api_url,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        dataset_name=dataset_name,
    )
    dataset_id = dataset_creation_response["id"]
    _ = await add_guardrail_to_dataset(
        explorer_api_url,
        dataset_id=dataset_id,
        policy='raise "pun detected in user message" if:\n   (msg: Message)\n   "pun" in msg.content and msg.role == "user"',
        action="block" if is_block_action else "log",
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
    )

    user_prompt = "Tell me a one sentence pun."
    request = {
        "model": "gemini-2.0-flash",
        "contents": user_prompt,
        "config": {
            "maxOutputTokens": 100,
        },
    }
    if is_block_action:
        if do_stream:
            chat_response = client.models.generate_content_stream(**request)

            assert_is_streamed_refusal(
                chat_response,
                [
                    "[Invariant] The request did not pass the guardrails",
                    "pun detected in user message",
                ],
            )
        else:
            with pytest.raises(genai.errors.ClientError) as exc_info:
                chat_response = client.models.generate_content(**request)
            assert "[Invariant] The request did not pass the guardrails" in str(
                exc_info.value
            )
            assert "pun detected in user message" in str(exc_info.value)
    else:
        if do_stream:
            response = client.models.generate_content_stream(**request)
            for _ in response:
                pass
        else:
            _ = client.models.generate_content(**request)

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

    assert len(trace["messages"]) == 2 if not is_block_action else 1
    assert trace["messages"][0] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_prompt,
            }
        ],
    }
    if not is_block_action:
        assert trace["messages"][1].get("role") == "assistant"

    # Fetch annotations
    annotations_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}/annotations",
        timeout=5,
    )
    annotations = annotations_response.json()

    assert len(annotations) == 1
    assert (
        annotations[0]["content"] == "pun detected in user message"
        and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        and annotations[0]["extra_metadata"]["guardrail-action"] == "block"
        if is_block_action
        else "log"
    )


def is_refusal(chunk):
    return (
        len(chunk.candidates) == 1
        and chunk.candidates[0].content.parts[0].text.startswith("[Invariant]")
        and chunk.prompt_feedback is not None
        and "BlockedReason.SAFETY" in str(chunk.prompt_feedback)
    )


def assert_is_streamed_refusal(response, expected_message_components: list[str]):
    """
    Validates that the streamed response contains a refusal at the end (or as only message).
    """
    num_chunks = 0
    for c in response:
        num_chunks += 1

    assert num_chunks >= 1, "Expected at least one chunk"
    # last chunk must be a refusal
    assert is_refusal(c)

    for emc in expected_message_components:
        assert (
            emc in c.model_dump_json()
        ), f"Expected message component {emc} not found in refusal message: {c.model_dump_json()}"

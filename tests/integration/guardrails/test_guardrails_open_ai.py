"""Test the guardrails from file with the OpenAI route."""

import os
import sys
import time
import uuid

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from openai import APIError, BadRequestError
from utils import add_guardrail_to_dataset, create_dataset, get_open_ai_client

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
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

    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"
    client = get_open_ai_client(gateway_url, push_to_explorer, dataset_name)

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of Spain?"}],
    }

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=False,
            )

        assert exc_info.value.status_code == 400
        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        assert "Madrid detected in the response" in str(exc_info.value)

    else:
        with pytest.raises(APIError) as exc_info:
            chat_response = client.chat.completions.create(
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

        assert len(annotations) == 1
        assert (
            annotations[0]["content"] == "Madrid detected in the response"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Get the capital for a given country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country_name": {
                            "type": "string",
                            "description": "The country name example Germany or France.",
                        }
                    },
                    "required": ["country_name"],
                },
            },
        }
    ]
    system_message = {
        "role": "system",
        "content": "Use the get_capital tool call to get the capital of a country. If the user input doesn't contain a country name, fail the request with a pretty message. If the get_capital tool call returns 'not_found' then fail the request with a pretty message. Do not return the capital if the get_capital tool call returns 'not_found'.",
    }
    request = {
        "model": "gpt-4o",
        "messages": [
            system_message,
            {"role": "user", "content": "What is the capital of Germany?"},
        ],
        "tools": tools,
    }

    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"
    client = get_open_ai_client(gateway_url, push_to_explorer, dataset_name)

    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=False,
            )

        assert exc_info.value.status_code == 400
        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        assert "get_capital is called with Germany as argument" in str(exc_info.value)

    else:
        with pytest.raises(APIError) as exc_info:
            chat_response = client.chat.completions.create(
                **request,
                stream=True,
            )

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

        assert len(trace["messages"]) == 3
        assert trace["messages"][0] == system_message
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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_input_from_guardrail_from_file(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the message content guardrail."""
    if not os.getenv("INVARIANT_API_KEY"):
        pytest.fail("No INVARIANT_API_KEY set, failing")

    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"
    client = get_open_ai_client(gateway_url, push_to_explorer, dataset_name)

    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me more about Fight Club."}],
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
        assert "Users must not mention the magic phrase 'Fight Club'" in str(
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
        assert "Users must not mention the magic phrase 'Fight Club'" in str(
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
            "content": "Tell me more about Fight Club.",
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
            == "Users must not mention the magic phrase 'Fight Club'"
            and annotations[0]["extra_metadata"]["source"] == "guardrails-error"
        )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize("do_stream", [True, False])
async def test_with_guardrails_from_explorer(explorer_api_url, gateway_url, do_stream):
    """Test that the guardrails from the explorer work."""
    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"
    client = get_open_ai_client(
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
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of Spain?"}],
        "max_tokens": 100,
    }
    if not do_stream:
        chat_response = client.chat.completions.create(
            **spain_request,
            stream=False,
        )

        assert "Madrid" in chat_response.choices[0].message.content
    else:
        chat_response = client.chat.completions.create(
            **spain_request,
            stream=True,
        )

        merged_content = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                merged_content += chunk.choices[0].delta.content
        assert "Madrid" in merged_content

    # Ask about Shrek
    # This should be blocked by the guardrails from the explorer
    user_prompt = "What kind of a creature is Shrek? What is his Shrek's wife's name? Only answer these questions with single sentences, don't add any extra details."
    shrek_request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "max_tokens": 100,
    }
    if not do_stream:
        with pytest.raises(BadRequestError) as exc_info:
            chat_response = client.chat.completions.create(
                **shrek_request,
                stream=False,
            )

        assert exc_info.value.status_code == 400
        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
        )
        # Only the block guardrail should be triggered here
        assert "ogre detected in response" in str(exc_info.value)
        assert "Fiona detected in response" not in str(exc_info.value)
    else:
        with pytest.raises(APIError) as exc_info:
            chat_response = client.chat.completions.create(
                **shrek_request,
                stream=True,
            )
            for _ in chat_response:
                pass

        assert "[Invariant] The response did not pass the guardrails" in str(
            exc_info.value
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
        "content": user_prompt,
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
        and annotations[0]["extra_metadata"]["guardrail"]["action"] == "block"
    )
    assert (
        annotations[1]["content"] == "Fiona detected in response"
        and annotations[1]["extra_metadata"]["source"] == "guardrails-error"
        and annotations[1]["extra_metadata"]["guardrail"]["action"] == "log"
    )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, is_block_action",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_preguardrailing_with_guardrails_from_explorer(
    explorer_api_url, gateway_url, do_stream, is_block_action
):
    """Test that the guardrails from the explorer work."""
    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"
    client = get_open_ai_client(
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
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "max_tokens": 100,
    }
    if is_block_action:
        if do_stream:
            with pytest.raises(APIError) as exc_info:
                chat_response = client.chat.completions.create(
                    **request,
                    stream=True,
                )
                for _ in chat_response:
                    pass

            assert "[Invariant] The request did not pass the guardrails" in str(
                exc_info.value
            )
        else:
            with pytest.raises(BadRequestError) as exc_info:
                chat_response = client.chat.completions.create(
                    **request,
                    stream=False,
                )

            assert exc_info.value.status_code == 400
            assert "[Invariant] The request did not pass the guardrails" in str(
                exc_info.value
            )
            assert "pun detected in user message" in str(exc_info.value)
    else:
        if do_stream:
            response = client.chat.completions.create(
                **request,
                stream=True,
            )
            for _ in response:
                pass
        else:
            _ = client.chat.completions.create(
                **request,
                stream=False,
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
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    # Fetch the trace
    trace_response = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id}",
        timeout=5,
    )
    trace = trace_response.json()

    assert len(trace["messages"]) == 1 if is_block_action else 2
    assert trace["messages"][0] == {
        "role": "user",
        "content": user_prompt,
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
        and annotations[0]["extra_metadata"]["guardrail"]["action"] == "block"
        if is_block_action
        else "log"
    )

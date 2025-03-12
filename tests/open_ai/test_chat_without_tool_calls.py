"""Test the chat completions gateway calls without tool calling."""

import base64
import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

# Add tests folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from httpx import Client
from openai import NotFoundError, OpenAI

from util import *  # Needed for pytest fixtures

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_chat_completion(
    context, explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the chat completions gateway calls without tool calling."""
    dataset_name = "test-dataset-open-ai-" + str(uuid.uuid4())

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=do_stream,
    )

    # Verify the chat response
    if not do_stream:
        assert "PARIS" in chat_response.choices[0].message.content.upper()
        expected_assistant_message = chat_response.choices[0].message.content
    else:
        full_response = ""
        for chunk in chat_response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        assert "PARIS" in full_response.upper()
        expected_assistant_message = full_response

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
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

        for message in trace["messages"]:
            message.pop("annotations", None)

        # Verify the trace messages
        assert trace["messages"] == [
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
            {
                "role": "assistant",
                "content": expected_assistant_message,
            },
        ]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize("push_to_explorer", [True, False])
async def test_chat_completion_with_image(
    context, explorer_api_url, gateway_url, push_to_explorer
):
    """Test the chat completions gateway works with image."""
    dataset_name = "test-dataset-open-ai-" + str(uuid.uuid4())

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )
    image_path = Path(__file__).parent.parent / "images" / "two-cats.png"
    with image_path.open("rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many cats are there in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )

        assert (
            "TWO" in chat_response.choices[0].message.content.upper()
            or "2" in chat_response.choices[0].message.content
        )

        if push_to_explorer:
            # Wait for the trace to be saved
            # This is needed because the trace is saved asynchronously
            time.sleep(2)
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

            for message in trace["messages"]:
                message.pop("annotations", None)

            # Verify the trace messages
            assert trace["messages"] == [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many cats are there in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64_image
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": chat_response.choices[0].message.content,
                },
            ]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
async def test_chat_completion_with_invariant_key_in_openai_key_header(
    context, explorer_api_url, gateway_url
):
    """Test the chat completions gateway calls with the Invariant API Key in the OpenAI Key header."""
    dataset_name = "test-dataset-open-ai-" + str(uuid.uuid4())
    openai_api_key = os.getenv("OPENAI_API_KEY")
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": openai_api_key + ";invariant-auth=<not needed for test>"},
    ):
        client = OpenAI(
            http_client=Client(),
            base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai",
        )

        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            stream=False,
        )

        # Verify the chat response
        assert "PARIS" in chat_response.choices[0].message.content.upper()
        expected_assistant_message = chat_response.choices[0].message.content

        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)

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

        for message in trace["messages"]:
            message.pop("annotations", None)

        # Verify the trace messages
        assert trace["messages"] == [
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
            {
                "role": "assistant",
                "content": expected_assistant_message,
            },
        ]


@pytest.mark.skip(reason="Skipping this test: OpenAI error scenario")
@pytest.mark.parametrize("do_stream", [True, False])
async def test_chat_completion_with_openai_exception(gateway_url, do_stream):
    """Test the chat completions gateway call when OpenAI API fails."""

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{"test-dataset-open-ai-" + str(uuid.uuid4())}/openai",
    )

    with pytest.raises(Exception) as exc_info:
        _ = client.chat.completions.create(
            model="gpt-4-vision-preview",  # This model is not available so we get a 404 error
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many cats are there in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + "01234"},
                        },
                    ],
                }
            ],
            stream=do_stream,
        )

    assert exc_info.errisinstance(NotFoundError)
    assert exc_info.value.status_code == 404

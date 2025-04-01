"""Test the generate content gateway calls without tool calling."""

import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import PIL.Image
import requests
from google import genai

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_generate_content(
    explorer_api_url, gateway_url, do_stream, push_to_explorer
):
    """Test the generate content gateway calls without tool calling."""
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
            "headers": {
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },  # This key is not used for local tests
        },
    )
    request = {
        "model": "gemini-2.0-flash",
        "contents": "What is the capital of France?",
        "config": {
            "maxOutputTokens": 100,
            "system_instruction": "This is the system instruction.",
        },
    }

    chat_response = (
        client.models.generate_content(**request)
        if not do_stream
        else client.models.generate_content_stream(**request)
    )

    # Verify the chat response
    if not do_stream:
        assert "PARIS" in chat_response.candidates[0].content.parts[0].text.upper()
        expected_assistant_message = chat_response.candidates[0].content.parts[0].text
    else:
        full_response = ""
        for chunk in chat_response:
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                for text_part in chunk.candidates[0].content.parts:
                    full_response += text_part.text
        assert "PARIS" in full_response.upper()
        expected_assistant_message = full_response

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
            f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
        )
        trace = trace_response.json()

        # Verify the trace messages
        assert trace["messages"] == [
            {
                "role": "system",
                "content": "This is the system instruction.",
            },
            {
                "role": "user",
                "content": [{"text": "What is the capital of France?", "type": "text"}],
            },
            {
                "role": "assistant",
                "content": expected_assistant_message,
            },
        ]


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize("push_to_explorer", [True, False])
async def test_generate_content_with_image(
    explorer_api_url, gateway_url, push_to_explorer
):
    """Test that generate content gateway calls work with image."""
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
            "headers": {
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },  # This key is not used for local tests
        },
    )

    image_path = Path(__file__).parent.parent / "resources" / "images" / "two-cats.png"
    image = PIL.Image.open(image_path)

    chat_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["How many cats are there in this image?", image],
        config={"maxOutputTokens": 100},
    )

    assert (
        "TWO" in chat_response.candidates[0].content.parts[0].text.upper()
        or "2" in chat_response.candidates[0].content.parts[0].text
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
            f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
        )
        trace = trace_response.json()
        # Verify the trace messages
        assert len(trace["messages"]) == 2
        assert trace["messages"][0]["role"] == "user"
        assert trace["messages"][0]["content"][0] == {
            "type": "text",
            "text": "How many cats are there in this image?",
        }
        assert trace["messages"][0]["content"][1]["type"] == "image_url"
        assert trace["messages"][1] == {
            "role": "assistant",
            "content": chat_response.candidates[0].content.parts[0].text,
        }


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
async def test_generate_content_with_invariant_key_in_gemini_key_header(
    explorer_api_url, gateway_url
):
    """Test the generate content gateway calls with the Invariant API Key in the Gemini Key header."""
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    invariant_key_suffix = f";invariant-auth={os.getenv('INVARIANT_API_KEY')}"
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": gemini_api_key + invariant_key_suffix},
    ):
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={
                "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            },
        )

        chat_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="What is the capital of Spain?",
            config={
                "maxOutputTokens": 100,
            },
        )

        # Verify the chat response
        assert "MADRID" in chat_response.candidates[0].content.parts[0].text.upper()
        expected_assistant_message = chat_response.candidates[0].content.parts[0].text

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

        # Verify the trace messages
        assert trace["messages"] == [
            {
                "role": "user",
                "content": [{"text": "What is the capital of Spain?", "type": "text"}],
            },
            {
                "role": "assistant",
                "content": expected_assistant_message,
            },
        ]

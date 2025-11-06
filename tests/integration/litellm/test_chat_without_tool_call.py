"""Test the chat completions gateway calls with tool calling through litellm."""

import os
import time
import uuid

import pytest
import requests
from litellm import completion

MODEL_API_KEYS = {
    "openai/gpt-4o": "OPENAI_API_KEY",
    "gemini/gemini-2.5-flash-preview-09-2025": "GEMINI_API_KEY",
    "anthropic/claude-3-5-haiku-20241022": "ANTHROPIC_API_KEY",
}


@pytest.mark.parametrize(
    "litellm_model",
    MODEL_API_KEYS.keys(),
)
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(False, False)],
)
async def test_chat_completion(
    explorer_api_url: str,
    litellm_model: str,
    gateway_url: str,
    do_stream: bool,
    push_to_explorer: bool,
):
    """Test the chat completions gateway calls with tool calling through litellm."""
    # Check if the API key is set in the environment variables
    api_key_env_var = MODEL_API_KEYS[litellm_model]
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        pytest.skip(f"Skipping {litellm_model} because {api_key_env_var} is not set")

    dataset_name = f"test-dataset-litellm-{litellm_model}-{uuid.uuid4()}"
    base_url = (
        f"{gateway_url}/api/v1/gateway/{dataset_name}"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway"
    )

    base_url += "/" + litellm_model.split("/")[0]  # add provider name
    if litellm_model.split("/")[0] == "gemini":
        base_url += f"/v1beta/models/{litellm_model.split('/')[1]}"  # gemini expects the model name in the url

    chat_response = completion(
        model=litellm_model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        extra_headers={
            "Invariant-Authorization": f"Bearer {os.environ['INVARIANT_API_KEY']}"
        },
        stream=do_stream,
        base_url=base_url,
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

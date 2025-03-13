"""Test the chat completions gateway calls with tool calling and processing response."""

import os
import sys
import time
import uuid

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from google import genai
from google.genai import types

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`,
        `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {
        "brightness": brightness,
        "colorTemperature": color_temp,
    }


SYSTEM_INSTRUCTION = "This the system instruction. Use the function call."
USER_PROMPT = (
    "Turn the light to 50% brightness and set the color temperature to daylight. \
    Once you are done, respond with 'DONE'."
)
SET_LIGHT_VALUES_TOOL_CALL_ARGS = {"brightness": 50, "color_temp": "daylight"}
SET_LIGHT_VALUES_TOOL_CALL = {
    "type": "function",
    "function": {
        "name": "set_light_values",
        "arguments": SET_LIGHT_VALUES_TOOL_CALL_ARGS,
    },
}


def _verify_trace_from_explorer(
    explorer_api_url, dataset_name, expected_final_assistant_message
) -> None:
    # Fetch the trace ids for the dataset.
    # There will be 2 traces - the first will contain the system instruction, user prompt
    # and the assistant tool call.
    # The second will contain the system instruction, user prompt, the assistant tool call,
    # the tool response and the assistant response.
    traces_response = requests.get(
        f"{explorer_api_url}/api/v1/dataset/byuser/developer/{dataset_name}/traces",
        timeout=5,
    )
    traces = traces_response.json()
    assert len(traces) == 2
    trace_id_1 = traces[0]["id"]
    trace_id_2 = traces[1]["id"]

    # Fetch the trace
    trace_response_1 = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id_1}",
        timeout=5,
    )
    trace_1 = trace_response_1.json()

    trace_response_2 = requests.get(
        f"{explorer_api_url}/api/v1/trace/{trace_id_2}",
        timeout=5,
    )
    trace_2 = trace_response_2.json()

    # Verify the trace messages
    assert trace_1["messages"] == [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT,
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [SET_LIGHT_VALUES_TOOL_CALL],
        },
    ]

    assert trace_2["messages"] == [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT,
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [SET_LIGHT_VALUES_TOOL_CALL],
        },
        {
            "role": "tool",
            "tool_name": "set_light_values",
            "content": {
                "brightness": 50,
                "colorTemperature": "daylight",
            },
        },
        {"role": "assistant", "content": expected_final_assistant_message},
    ]


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
@pytest.mark.parametrize(
    "do_stream, push_to_explorer",
    [(True, True), (True, False), (False, True), (False, False)],
)
async def test_generate_content_with_tool_call(
    explorer_api_url, gateway_url, push_to_explorer, do_stream
):
    """
    Test the generate content gateway calls with tool calling and response processing
    without streaming.
    """
    dataset_name = f"test-dataset-gemini-{uuid.uuid4()}"

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
            "headers": {
                "invariant-authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        },
    )

    request = {
        "model": "gemini-2.0-flash",
        "contents": USER_PROMPT,
        "config": types.GenerateContentConfig(
            tools=[set_light_values],
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    }

    chat_response = (
        client.models.generate_content(**request)
        if not do_stream
        else client.models.generate_content_stream(**request)
    )

    if not do_stream:
        assert "DONE" in chat_response.candidates[0].content.parts[0].text
        expected_final_assistant_message = (
            chat_response.candidates[0].content.parts[0].text
        )
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
        assert "DONE" in full_response.upper()
        expected_final_assistant_message = full_response

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
        _verify_trace_from_explorer(
            explorer_api_url, dataset_name, expected_final_assistant_message
        )

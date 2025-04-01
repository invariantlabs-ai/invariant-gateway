"""Test the chat completions gateway calls with tool calling and processing response."""

import json
import os
import sys
import time
import uuid

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from httpx import Client
from openai import OpenAI

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_chat_completion_with_tool_call_without_streaming(
    explorer_api_url, gateway_url, push_to_explorer
):
    """
    Test the chat completions gateway calls with tool calling and response processing
    without streaming.
    """
    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },  # This key is not used for local tests
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the weather in New York?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
    )

    assert chat_response.choices[0].message.role == "assistant"
    # Extract tool call
    assert len(chat_response.choices[0].message.tool_calls) == 1
    tool_call = chat_response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert (
        "New York" in tool_call.function.arguments
        and "location" in tool_call.function.arguments
    )

    # Mock response of tool call
    tool_result = "The temperature in New York is 15°C and it is raining."
    history = [
        {"role": "user", "content": "What is the weather in New York?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": tool_call.function.arguments,
                        "name": tool_call.function.name,
                    },
                    "id": tool_call.id,
                    "type": tool_call.type,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "tool_name": "get_weather",
            "content": tool_result,
        },
    ]

    # Send mock response back to OpenAI with history of chat
    chat_response_final = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
    )
    assert "15°C" in chat_response_final.choices[0].message.content

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
        expected_messages = history + [
            {
                "role": "assistant",
                "content": chat_response_final.choices[0].message.content,
            }
        ]
        expected_messages[1]["tool_calls"][0]["function"]["arguments"] = json.loads(
            expected_messages[1]["tool_calls"][0]["function"]["arguments"]
        )
        assert trace["messages"] == expected_messages


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_chat_completion_with_tool_call_with_streaming(
    explorer_api_url, gateway_url, push_to_explorer
):
    """
    Test the chat completions gateway calls with tool calling and response processing
    while streaming.
    """
    dataset_name = f"test-dataset-open-ai-{uuid.uuid4()}"

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },  # This key is not used for local tests
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the weather in New York?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        stream=True,
    )

    tool_call = {"function": {}}
    for chunk in chat_response:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            partial_tool_call = chunk.choices[0].delta.tool_calls[0]
            tool_call.setdefault("id", partial_tool_call.id)
            tool_call.setdefault("type", partial_tool_call.type)
            tool_call["function"].setdefault("name", partial_tool_call.function.name)
            tool_call["function"].setdefault("arguments", "")
            tool_call["function"]["arguments"] += partial_tool_call.function.arguments

    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"location":"New York"}'

    # Mock response of tool call
    tool_result = "The temperature in New York is 15°C and it is raining."
    history = [
        {"role": "user", "content": "What is the weather in New York?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": tool_call["function"]["arguments"],
                        "name": tool_call["function"]["name"],
                    },
                    "id": tool_call["id"],
                    "type": tool_call["type"],
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "tool_name": "get_weather",
            "content": tool_result,
        },
    ]

    # Send mock response back to OpenAI with history of chat
    chat_response_final = client.chat.completions.create(
        model="gpt-4o", messages=history, stream=True
    )
    final_response = {"role": "assistant", "content": ""}

    for chunk in chat_response_final:
        if chunk.choices and chunk.choices[0].delta.content:
            final_response["content"] += chunk.choices[0].delta.content

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

        # Verify the trace messages
        expected_messages = history + [final_response]
        expected_messages[1]["tool_calls"][0]["function"]["arguments"] = json.loads(
            expected_messages[1]["tool_calls"][0]["function"]["arguments"]
        )
        assert trace["messages"] == expected_messages

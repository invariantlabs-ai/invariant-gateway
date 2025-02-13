"""Test the chat completions proxy calls with tool calling and processing response."""

import json
import os
import sys
import uuid

import pytest
from httpx import Client

# add tests folder (parent) to sys.path
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import *  # needed for pytest fixtures

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
async def test_chat_completion_with_tool_call_without_streaming(
    context, explorer_api_url, proxy_url
):
    """
    Test the chat completions proxy calls with tool calling and response processing
    without streaming.
    """
    dataset_name = "test-dataset-open-ai-tool-call-" + str(uuid.uuid4())

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{proxy_url}/api/v1/proxy/{dataset_name}/openai",
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
    assert chat_response.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert (
        chat_response.choices[0].message.tool_calls[0].function.arguments
        == '{"location":"New York"}'
    )
    tool_call = chat_response.choices[0].message.tool_calls[0]

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
async def test_chat_completion_with_tool_call_with_streaming(
    context, explorer_api_url, proxy_url
):
    """
    Test the chat completions proxy calls with tool calling and response processing
    while streaming.
    """
    dataset_name = "test-dataset-open-ai-tool-call-" + str(uuid.uuid4())

    client = OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer <some-key>"
            },  # This key is not used for local tests
        ),
        base_url=f"{proxy_url}/api/v1/proxy/{dataset_name}/openai",
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
    attempt = 0
    while attempt<3:
        try:
            for chunk in chat_response_final:
                if chunk.choices and chunk.choices[0].delta.content:
                    final_response["content"] += chunk.choices[0].delta.content
            break
        except httpx.RemoteProtocolError as e:
            attempt += 1
            print(f"Streming error on attempt {attempt}: {e}")
    else:
        print("Max retries reached. Exiting.")
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

    # Verify the trace messages
    expected_messages = history + [final_response]
    expected_messages[1]["tool_calls"][0]["function"]["arguments"] = json.loads(
        expected_messages[1]["tool_calls"][0]["function"]["arguments"]
    )
    assert trace["messages"] == expected_messages

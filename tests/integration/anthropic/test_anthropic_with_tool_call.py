"""Test the Anthropic messages API with tool call for the weather agent."""

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Add integration folder (parent) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
import pytest
import requests
from utils import get_anthropic_client

# Pytest plugins
pytest_plugins = ("pytest_asyncio",)


class WeatherAgent:
    """Weather agent to get the current weather in a given location."""

    def __init__(self, gateway_url, push_to_explorer):
        self.dataset_name = f"test-dataset-anthropic-{uuid.uuid4()}"
        self.client = get_anthropic_client(
            gateway_url, push_to_explorer, self.dataset_name
        )
        self.get_weather_function = {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit of temperature, either "celsius" or "fahrenheit"',
                    },
                },
                "required": ["location"],
            },
        }

    def get_response(self, messages: list[dict]) -> list[dict]:
        """
        Get the response from the agent for a given user query for weather.
        """
        response_list = []
        while True:
            response = self.client.messages.create(
                tools=[self.get_weather_function],
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages,
            )
            response_list.append(response)
            # If there's tool call, Extract the tool call parameters from the response
            if len(response.content) > 1 and response.content[1].type == "tool_use":
                tool_call_params = response.content[1].input
                tool_call_result = self.get_weather(tool_call_params["location"])
                tool_call_id = response.content[1].id
                messages.append({"role": response.role, "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": tool_call_result,
                            }
                        ],
                    }
                )
            else:
                return response_list

    def get_streaming_response(self, messages: list[dict]) -> list[dict]:
        """Get streaming response from the agent for a given user query for weather."""
        response_list = []

        def clean_quotes(text):
            # Convert \' to '
            text = text.replace("'", "'")
            # Convert \" to "
            text = text.replace('"', '"')
            text = text.replace("\n", " ")
            return text

        while True:
            json_data = ""
            content = []
            event = None
            with self.client.messages.stream(
                tools=[self.get_weather_function],
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages,
            ) as stream:
                for event in stream:
                    if isinstance(event, anthropic.types.RawContentBlockStartEvent):
                        # Start a new block
                        current_block = event.content_block
                        current_text = ""
                    elif isinstance(event, anthropic.types.RawContentBlockDeltaEvent):
                        if hasattr(event.delta, "text"):
                            # Accumulate text for TextBlock
                            current_text += clean_quotes(event.delta.text)
                        elif hasattr(event.delta, "partial_json"):
                            # Accumulate JSON for ToolUseBlock
                            json_data += clean_quotes(event.delta.partial_json)
                            current_text += clean_quotes(event.delta.partial_json)
                    elif isinstance(event, anthropic.types.RawContentBlockStopEvent):
                        # Block is complete, add it to content
                        if current_block.type == "text":
                            content.append(
                                anthropic.types.TextBlock(
                                    citations=None, text=current_text, type="text"
                                )
                            )
                        elif current_block.type == "tool_use":
                            content.append(
                                anthropic.types.ToolUseBlock(
                                    id=current_block.id,
                                    input=json.loads(current_text),
                                    name=current_block.name,
                                    type="tool_use",
                                )
                            )
            if content:
                response_list.append(content)
            if (
                isinstance(event, anthropic.types.RawMessageStopEvent)
                and event.message.stop_reason == "tool_use"
            ):
                tool_call_params = json.loads(json_data)
                tool_call_result = self.get_weather(tool_call_params["location"])
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content[-1].id,
                                "content": tool_call_result,
                            }
                        ],
                    }
                )
            else:
                return response_list

    def get_weather(self, location: str):
        """Get the current weather in a given location using latitude and longitude."""
        response = f"""Weather in {location}:
                    Good morning! Expect overcast skies with intermittent showers throughout the day. 
                    Temperatures will range from a cool 15°C in the early hours to around 19°C by mid-afternoon.
                    Light winds from the northeast at about 10 km/h will keep conditions mild. 
                    It might be a good idea to carry an umbrella if you’re heading out. Stay dry and have a great day!
                    """
        return response


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_response_with_tool_call(explorer_api_url, gateway_url, push_to_explorer):
    """Test the chat completion without streaming for the weather agent."""

    weather_agent = WeatherAgent(gateway_url, push_to_explorer)

    query = "Tell me the weather for New York in Celsius"

    city = "new york"
    # Process each query
    responses = []
    messages = [{"role": "user", "content": query}]
    response = weather_agent.get_response(messages)
    assert response is not None
    assert response[0].role == "assistant"
    assert response[0].stop_reason == "tool_use"
    assert response[0].content[0].type == "text"
    assert response[0].content[1].type == "tool_use"
    assert city in response[0].content[1].input["location"].lower()

    assert response[1].role == "assistant"
    assert response[1].stop_reason == "end_turn"
    responses.append(response)

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{weather_agent.dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()
        trace = traces[-1]
        trace_id = trace["id"]
        # Fetch the trace
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
        )
        trace = trace_response.json()
        trace_messages = trace["messages"]

        assert trace_messages[0]["role"] == "user"
        assert trace_messages[0]["content"] == query
        assert trace_messages[1]["role"] == "assistant"
        assert city in trace_messages[1]["content"].lower()
        assert trace_messages[2]["role"] == "assistant"
        assert trace_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert (
            city
            in trace_messages[2]["tool_calls"][0]["function"]["arguments"][
                "location"
            ].lower()
        )
        assert trace_messages[3]["role"] == "tool"
        assert trace_messages[4]["role"] == "assistant"
        assert city in trace_messages[4]["content"].lower()


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_streaming_response_with_tool_call(
    explorer_api_url, gateway_url, push_to_explorer
):
    """Test the chat completion with streaming for the weather agent."""
    weather_agent = WeatherAgent(gateway_url, push_to_explorer)

    query = "Tell me the weather for New York in Celsius"
    city = "new york"

    messages = [{"role": "user", "content": query}]
    response = weather_agent.get_streaming_response(messages)

    if len(response) == 2:
        assert response is not None
        assert response[0][0].type == "text"
        assert response[0][1].type == "tool_use"
        assert response[0][1].name == "get_weather"
        assert city in response[0][1].input["location"].lower()

        assert response[1][0].type == "text"
        assert city in response[1][0].text.lower()
    elif len(response) == 1:
        # expected output in this case is something like this:
        # [[TextBlock(text="I'll help you check the weather in New York using the get_weather function.", type='text', citations=None), ToolUseBlock(id='toolu_019VZsmxuUhShou2EpPBxvpe', input={'location': 'New York, NY', 'unit': 'celsius'}, name='get_weather', type='tool_use')]]

        assert response is not None
        assert response[0][0].type == "text"
        assert response[0][1].type == "tool_use"
        assert response[0][1].name == "get_weather"
        assert city in response[0][1].input["location"].lower()

    else:
        assert False, "Expected response length 2 or 1, but got" + str(response)

    if push_to_explorer:
        # Wait for the trace to be saved
        # This is needed because the trace is saved asynchronously
        time.sleep(2)
        traces_response = requests.get(
            f"{explorer_api_url}/api/v1/dataset/byuser/developer/{weather_agent.dataset_name}/traces",
            timeout=5,
        )
        traces = traces_response.json()

        trace = traces[-1]
        trace_id = trace["id"]
        # Fetch the trace
        trace_response = requests.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
        )
        trace = trace_response.json()
        trace_messages = trace["messages"]
        assert trace_messages[0]["role"] == "user"
        assert trace_messages[0]["content"] == query
        assert trace_messages[1]["role"] == "assistant"
        assert city in trace_messages[1]["content"].lower()
        assert trace_messages[2]["role"] == "assistant"
        assert trace_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert (
            city
            in trace_messages[2]["tool_calls"][0]["function"]["arguments"][
                "location"
            ].lower()
        )
        assert trace_messages[3]["role"] == "tool"
        assert trace_messages[4]["role"] == "assistant"
        assert city in trace_messages[4]["content"].lower()


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set"
)
@pytest.mark.parametrize("push_to_explorer", [False, True])
async def test_response_with_tool_call_with_image(
    explorer_api_url, gateway_url, push_to_explorer
):
    """Test the chat completion with image for the weather agent."""
    weather_agent = WeatherAgent(gateway_url, push_to_explorer)

    image_path = Path(__file__).parent.parent / "resources" / "images" / "new-york.jpg"

    with image_path.open("rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        query = "get the weather in the city of this image in Celsius"
        city = "new york"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ]
        response = weather_agent.get_response(messages)
        assert response is not None
        assert response[0].role == "assistant"
        assert response[0].stop_reason == "tool_use"
        assert response[0].content[0].type == "text"
        assert response[0].content[1].type == "tool_use"
        assert city in response[0].content[1].input["location"].lower()

        assert response[1].role == "assistant"
        assert response[1].stop_reason == "end_turn"

        if push_to_explorer:
            # Wait for the trace to be saved
            # This is needed because the trace is saved asynchronously
            time.sleep(2)
            traces_response = requests.get(
                f"{explorer_api_url}/api/v1/dataset/byuser/developer/{weather_agent.dataset_name}/traces",
                timeout=5,
            )
            traces = traces_response.json()

            trace = traces[-1]
            trace_id = trace["id"]
            trace_response = requests.get(
                f"{explorer_api_url}/api/v1/trace/{trace_id}", timeout=5
            )
            trace = trace_response.json()
            trace_messages = trace["messages"]
            assert trace_messages[0]["role"] == "user"
            assert trace_messages[1]["role"] == "assistant"
            assert city in trace_messages[1]["content"].lower()
            assert trace_messages[2]["role"] == "assistant"
            assert (
                trace_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
            )
            assert (
                city
                in trace_messages[2]["tool_calls"][0]["function"]["arguments"][
                    "location"
                ].lower()
            )
            assert trace_messages[3]["role"] == "tool"

import datetime
import os
from typing import Dict
import json
import anthropic
import pytest
from httpx import Client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import *  # needed for pytest fixtures

pytest_plugins = ("pytest_asyncio",)


class WeatherAgent:
    def __init__(self,proxy_url):
        self.dataset_name = "claude_weather_agent_test" + str(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        invariant_api_key = os.environ.get("INVARIANT_API_KEY","None")
        self.client = anthropic.Anthropic(
            http_client=Client(
                headers={"Invariant-Authorization": f"Bearer {invariant_api_key}"},
            ),
            base_url=f"{proxy_url}/api/v1/proxy/{self.dataset_name}/anthropic",
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

    def get_response(self, user_query: str) -> Dict:
        """
        Get the response from the agent for a given user query for weather.
        """
        messages = [{"role": "user", "content": user_query}]
        response_list = []
        while True:
            response = self.client.messages.create(
                tools=[self.get_weather_function],
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
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
            
    def get_streaming_response(self, user_query: str) -> Dict:
        messages = [{"role": "user", "content": user_query}]
        response_list = []

        def clean_quotes(text):
            # Convert \' to '
            text = text.replace("\'", "'")
            # Convert \" to "
            text = text.replace('\"', '"')
            text = text.replace("\n", " ")
            return text
        
        while True:
            json_data = ""
            content = []
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
                        if hasattr(event.delta, 'text'):
                            # Accumulate text for TextBlock
                            current_text += clean_quotes(event.delta.text)
                        elif hasattr(event.delta, 'partial_json'):
                            # Accumulate JSON for ToolUseBlock
                            json_data += clean_quotes(event.delta.partial_json)
                            current_text += clean_quotes(event.delta.partial_json)
                    elif isinstance(event, anthropic.types.RawContentBlockStopEvent):
                        # Block is complete, add it to content
                        if current_block.type == 'text':
                            content.append(anthropic.types.TextBlock(citations=None, text=current_text, type="text"))
                        elif current_block.type == 'tool_use':
                            content.append(
                                anthropic.types.ToolUseBlock(id=current_block.id,
                                input=json.loads(current_text), 
                                name=current_block.name, 
                                type="tool_use")
                            )
            response_list.append(content)    
            if isinstance(event, anthropic.types.RawMessageStopEvent) and event.message.stop_reason == "tool_use":
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
        response = f'''Weather in {location}:
                    Good morning! Expect overcast skies with intermittent showers throughout the day. 
                    Temperatures will range from a cool 15°C in the early hours to around 19°C by mid-afternoon.
                    Light winds from the northeast at about 10 km/h will keep conditions mild. 
                    It might be a good idea to carry an umbrella if you’re heading out. Stay dry and have a great day!
                    '''
        return response

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
async def test_response_with_toolcall(
    context, explorer_api_url, proxy_url
):
    """Test the chat completion without streaming for the weather agent."""
    
    weather_agent = WeatherAgent(proxy_url)

    queries = [
        "What's the weather like in Zurich city?",
        "Tell me the weather for New York",
    ]
    cities = ["zurich", "new york"]
    # Process each query
    responses = []
    for index, query in enumerate(queries):
        response = weather_agent.get_response(query)
        assert response is not None
        assert response[0].role == "assistant"
        assert response[0].stop_reason == "tool_use"
        assert response[0].content[0].type == "text"
        assert response[0].content[1].type == "tool_use"
        assert cities[index] in response[0].content[1].input["location"].lower()

        assert response[1].role == "assistant"
        assert response[1].stop_reason == "end_turn"
        assert cities[index] in response[1].content[0].text.lower()
        responses.append(response)

    traces_response = await context.request.get(
    f"{explorer_api_url}/api/v1/dataset/byuser/developer/{weather_agent.dataset_name}/traces"
    )
    traces = await traces_response.json()
    assert len(traces) == len(queries)

    for index,trace in enumerate(traces): 
        trace_id = trace["id"]
        # Fetch the trace
        trace_response = await context.request.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}"
        )
        trace = await trace_response.json()
        trace_messages = trace["messages"]

        assert trace_messages[0]["role"] == "user"
        assert trace_messages[0]["content"] == queries[index]
        assert trace_messages[1]["role"] == "assistant"
        assert cities[index] in trace_messages[1]["content"].lower()
        assert trace_messages[2]["role"] == "assistant"
        assert trace_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert cities[index] in trace_messages[2]["tool_calls"][0]["function"]["arguments"]["location"].lower()
        assert trace_messages[3]["role"] == "tool"
        assert trace_messages[4]["role"] == "assistant"
        assert cities[index] in trace_messages[4]["content"].lower()



@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
async def test_streaming_response_with_toolcall(
    context, explorer_api_url, proxy_url
):
    """Test the chat completion with streaming for the weather agent."""
    weather_agent = WeatherAgent(proxy_url)

    queries = [
        "What's the weather like in Zurich city?",
        "Tell me the weather for New York",
    ]
    cities = ["zurich", "new york"]


    for index, query in enumerate(queries):
        response = weather_agent.get_streaming_response(query)
        assert response is not None
        assert response[0][0].type == "text"
        assert response[0][1].type == "tool_use"
        assert response[0][1].name == "get_weather"
        assert cities[index] in response[0][1].input["location"].lower()
        
        assert response[1][0].type == "text"
        assert cities[index] in response[1][0].text.lower()
    
    traces_response = await context.request.get(
    f"{explorer_api_url}/api/v1/dataset/byuser/developer/{weather_agent.dataset_name}/traces"
    )
    traces = await traces_response.json()
    assert len(traces) == len(queries)

    for index,trace in enumerate(traces): 
        trace_id = trace["id"]
        # Fetch the trace
        trace_response = await context.request.get(
            f"{explorer_api_url}/api/v1/trace/{trace_id}"
        )
        trace = await trace_response.json()
        trace_messages = trace["messages"]
        assert trace_messages[0]["role"] == "user"
        assert trace_messages[0]["content"] == queries[index]
        assert trace_messages[1]["role"] == "assistant"
        assert cities[index] in trace_messages[1]["content"].lower()
        assert trace_messages[2]["role"] == "assistant"
        assert trace_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert cities[index] in trace_messages[2]["tool_calls"][0]["function"]["arguments"]["location"].lower()
        assert trace_messages[3]["role"] == "tool"
        assert trace_messages[4]["role"] == "assistant"
        assert cities[index] in trace_messages[4]["content"].lower()


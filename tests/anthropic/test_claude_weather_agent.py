import datetime
import os
from typing import Dict

import anthropic
import pytest
from httpx import Client
from tavily import TavilyClient


class WeatherAgent:
    def __init__(self, api_key: str):
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        dataset_name = "claude_weather_agent_test" + str(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self.client = anthropic.Anthropic(
            http_client=Client(
                headers={"Invariant-Authorization": "Bearer <some-api-key>"},
            ),
            base_url=f"http://localhost/api/v1/proxy/{dataset_name}/anthropic",
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

        # self.system_prompt = """You are an assistant that can perform weather searches using function calls.
        #     When a user asks for weather information, respond with a JSON object specifying
        #     the function name `get_weather` and the arguments latitude and longitude are needed."""

    def get_response(self, user_query: str) -> Dict:
        """
        Get the response from the agent for a given user query for weather.
        """
        messages = [{"role": "user", "content": user_query}]
        while True:
            response = self.client.messages.create(
                # system=self.system_prompt,
                tools=[self.get_weather_function],
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages,
            )

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
                return response.content[0].text

    def get_weather(self, location: str):
        """Get the current weather in a given location using latitude and longitude."""
        query = f"What is the weather in {location}?"
        response = self.tavily_client.search(query)
        response_content = response["results"][0]["content"]
        return response["results"][0]["title"] + ":\n" + response_content


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("TAVILY_API_KEY"),
    reason="API keys not set",
)
def test_proxy_response():
    """Test the proxy response for the weather agent."""
    # Initialize agent with Anthropic API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    weather_agent = WeatherAgent(anthropic_api_key)

    # Example queries
    queries = [
        "What's the weather like in Zurich city?",
        "Tell me the forecast for New York",
        "How's the weather in London next week?",
    ]
    cities = ["Zurich", "New York", "London"]
    # Process each query
    for index, query in enumerate(queries):
        response = weather_agent.get_response(query)
        assert response is not None
        assert cities[index] in response

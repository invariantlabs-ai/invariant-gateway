import datetime
import os
from typing import Dict

import anthropic
import pytest
from httpx import Client


class WeatherAgent:
    def __init__(self):
        dataset_name = "claude_weather_agent_test" + str(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        invariant_api_key = os.environ.get("INVARIANT_API_KEY")
        self.client = anthropic.Anthropic(
            http_client=Client(
                headers={"Invariant-Authorization": f"Bearer {invariant_api_key}"},
            ),
            base_url=f"http://localhost/api/v1/proxy/{dataset_name}/anthropic",
        )
        self.get_weather_function = {
            "name": "get_weather",
            "description": "Get the current weather in a given locatiofn",
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
        response = f'''Weather in {location}:
                    Good morning! Expect overcast skies with intermittent showers throughout the day. Temperatures will range from a cool 15°C in the early hours to around 19°C by mid-afternoon. Light winds from the northeast at about 10 km/h will keep conditions mild. It might be a good idea to carry an umbrella if you’re heading out. Stay dry and have a great day!
                    '''
        return response


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),reason="Anthropic API keys not set")
def test_proxy_response():
    """Test the proxy response for the weather agent."""
    # Initialize agent with Anthropic API key
    weather_agent = WeatherAgent()

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
        print("response:",response)
        assert response is not None
        assert cities[index] in response

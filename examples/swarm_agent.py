from swarm import Swarm, Agent
from openai import OpenAI
from httpx import Client
import os

client = Swarm(
    client=OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": "Bearer "
                + os.getenv("INVARIANT_API_KEY", "")
            }
        ),
        base_url="https://explorer.invariantlabs.ai/api/v1/proxy/weather-swarm-agent/openai",
    )
)


def get_weather():
    return "It's sunny."


agent = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)

response = client.run(
    agent=agent,
    messages=[{"role": "user", "content": "What's the weather?"}],
)

print(response.messages[-1]["content"])

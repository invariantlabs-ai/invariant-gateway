"""
Simple (non-streaming) test client for the Gateway (uses OpenAI integration).
"""

from openai import OpenAI
from httpx import Client
import os

# unicode escape everything
guardrails = """
raise "Rule 1: Do not talk about Fight Club" if:
    (msg: Message)
    "fight club" in msg.content
""".encode("unicode_escape")

openai_client = OpenAI(
    default_headers={
        "Invariant-Authorization": "Bearer " + os.getenv("INVARIANT_API_KEY"),
        "Invariant-Guardrails": guardrails,
    },
    base_url="http://localhost:8005/api/v1/gateway/non-streaming/openai",
)

response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "What can you tell me about fight club?",
        }
    ],
)
print("Response: ", response.choices[0].message.content)

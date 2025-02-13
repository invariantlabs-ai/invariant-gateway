import anthropic
import os
import anthropic
from httpx import Client
import os
# from invariant import testing
import datetime
import pytest
import sys
import httpx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import *  # needed for pytest fixtures

pytest_plugins = ("pytest_asyncio")
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
def test_streaming_response_without_toolcall(proxy_url):    
    # Example queries
    dataset_name = "claude_streaming_agent_test" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    invariant_api_key = os.environ.get("INVARIANT_API_KEY")

    client = anthropic.Anthropic(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {invariant_api_key}"
            },
    ),
    base_url=f"{proxy_url}/api/v1/proxy/{dataset_name}/anthropic",
    )

    cities = ["zurich", "new york", "london"]
    queries = [
        "Can you introduce Zurich city within 200 words?",
        "Tell me the history of New York within 100 words?",
        "How's the weather in London next week?"
    ]
    # Process each query
    for index,query in enumerate(queries):
        messages = [
        {
            "role": "user",
            "content": query
        }
        ]
        response_text = ""
        attempt = 0
        while attempt<3:
            try:
                with client.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=messages,
                    # stream = True
                ) as response:
                    for reply in response.text_stream:
                        response_text += reply
                    assert cities[index] in response_text.lower()
                break
            except httpx.RemoteProtocolError as e:
                attempt += 1
                print(f"Streming error on attempt {attempt}: {e}")
        else:
            print("Max retries reached. Exiting.")
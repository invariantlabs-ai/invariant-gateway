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

@pytest.fixture
def client(proxy_url):
    dataset_name = "claude_streaming_response_without_toolcall_test" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    invariant_api_key = os.environ.get("INVARIANT_API_KEY","None")

    client = anthropic.Anthropic(
            http_client=Client(
                headers={"Invariant-Authorization": f"Bearer {invariant_api_key}"},
            ),
            base_url=f"{proxy_url}/api/v1/proxy/{dataset_name}/anthropic",
        )
    return client


pytest_plugins = ("pytest_asyncio")
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
async def test_chat_completion_without_streaming(client):

    cities = ["zurich", "new york", "london"]
    queries = [
        "Can you introduce Zurich city within 200 words?",
        "Tell me the history of New York within 100 words?",
        "How's the weather in London next week?"
    ]

    # Process each query
    for query in queries:
        response = client.messages.create(
                # system=self.system_prompt,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": query}],
            )
        response_text = response.content[0].text
        assert response_text is not None
        assert cities[queries.index(query)] in response_text.lower()

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
def test_streaming_response_without_toolcall(proxy_url,client):    

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
  
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages,
        ) as response:
            for reply in response.text_stream:
                response_text += reply
            assert cities[index] in response_text.lower()
    
        assert response_text is not None
        assert cities[queries.index(query)] in response_text.lower()

import anthropic
import os
import anthropic
from httpx import Client
import os
# from invariant import testing
import datetime
import pytest

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),reason="Anthropic API keys not set")
def test_streaming_response_without_toolcall():    
    # Example queries
    dataset_name = "claude_streaming_agent_test" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    invariant_api_key = os.environ.get("INVARIANT_API_KEY")

    client = anthropic.Anthropic(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {invariant_api_key}"
            },
    ),
    base_url=f"http://localhost/api/v1/proxy/{dataset_name}/anthropic",
    )

    cities = ["Zurich", "New York", "London"]
    queries = [
        "Can you introduce Zurich city within 200 words?",
        "Tell me the history of New York",
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
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages,
            # stream = True
        ) as response:
            for reply in response.text_stream:
                response_text += reply
                print(reply, end="", flush=True)
                assert reply != ""
            assert cities[index] in response_text
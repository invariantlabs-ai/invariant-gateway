# Explorer Proxy

This is a lightweight Docker service which sits between the user and the LLM Provider (OpenAI, Anthropic, etc) and pushes the resultant agent traces to the [Invariant Explorer](https://explorer.invariantlabs.ai/) which lets you visualize and explore traces.


## OpenAI

You can make requests to OpenAI while pushing traces to the Invariant Explorer.

First get an API key by following the steps [here](https://explorer.invariantlabs.ai/docs/explorer/Explorer_API/1_client_setup/).

Then you can use a custom `http_client` and `base_url` while setting up the OpenAI client object.

A dataset with the given name will be created if it already doesn't exist.

```python
from httpx import Client
from openai import OpenAI

client = OpenAI(
    http_client=Client(
        headers={
            "Invariant-Authorization": "Bearer <invariant-api-key>"
        },
    ),
    base_url="https://explorer.invariantlabs.ai/api/v1/proxy/<add-your-dataset-name-here>/openai",
)

# Make API requests to OpenAI as usual.
```

## Anthropic
Coming Soon!

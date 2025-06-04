# Invariant Gateway

**LLM proxy to observe and debug what your AI agents are doing.**

[Documentation](https://explorer.invariantlabs.ai/docs/gateway) | [Quickstart for Users](#quickstart-for-users) | [Quickstart for Developers](#quickstart-for-developers) | [Run Locally](#run-the-gateway-locally)

<a href="https://discord.gg/dZuZfhKnJ4"><img src="https://img.shields.io/discord/1265409784409231483?style=plastic&logo=discord&color=blueviolet&logoColor=white" height=18/></a>

Invariant Gateway is a lightweight _zero-configuration_ service that acts as an intermediary between AI Agents and LLM providers (such as OpenAI and Anthropic).

Gateway automatically traces agent interactions and stores them in the [Invariant Explorer](https://explorer.invariantlabs.ai/), giving you insights into what your agents are doing.
This allows you to _observe and debug_ your agents in [Invariant Explorer](https://explorer.invariantlabs.ai/).

<br/>
<br/>

<div align="center">
<img src="resources/images/overview.svg" alt="Invariant Gateway Diagram" width="80%"/>
</div>

<br/>
<br/>

- [x] **Single Line Setup**: Just change the base URL of your LLM provider to the Invariant Gateway.
- [x] **Intercepts agents on an LLM-level** for better debugging and analysis.
- [x] **Tool Calling and Computer Use Support** to capture all forms of agentic interactions.
- [x] **MCP Protocol Support** for both standard I/O, Server-Sent Events (SSE) and Streamable HTTP transports.
- [x] **Seamless forwarding and LLM streaming** to OpenAI, Anthropic, and other LLM providers.
- [x] **Store and organize runtime traces** in the [Invariant Explorer](https://explorer.invariantlabs.ai/).

## **Quickstart for Teams and Users**

Looking to observe and secure AI agents in your organization? See our [no-code quickstart guide for users](#quickstart-for-users) to get started.

## **Quickstart for Developers**

To add Gateway to your agentic system, follow one of the integration guides below, depending on the LLM provider.

## **Integration Guides**

### **🔹 OpenAI Integration**

Gateway supports the OpenAI Chat Completions API (`/v1/chat/completions` endpoint).

1. Follow [these steps](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key) to obtain an OpenAI API key.
2. **Modify OpenAI Client Setup**

   Instead of connecting directly to OpenAI, configure your `OpenAI` client to use Gateway:

   ```python
   from httpx import Client
   from openai import OpenAI

   client = OpenAI(
       http_client=Client(
           headers={
               "Invariant-Authorization": "Bearer your-invariant-api-key"
           },
       ),
       base_url="https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/openai",
   )
   ```

   > **Note:** Do not include the curly braces `{}`. If the dataset does not exist in Invariant Explorer, it will be created before adding traces.

### **🔹 Anthropic Integration**

Gateway supports the Anthropic Messages API (`/v1/messages` endpoint).

1. Follow [these steps](https://docs.anthropic.com/en/docs/initial-setup#set-your-api-key) to obtain an Anthropic API key.
2. **Modify Anthropic Client Setup**

   ```python
   from httpx import Client
   from anthropic import Anthropic

   client = Anthropic(
       http_client=Client(
           headers={
               "Invariant-Authorization": "Bearer your-invariant-api-key"
           },
       ),
       base_url="https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/anthropic",
   )
   ```

   > **Note:** Do not include the curly braces `{}`. If the dataset does not exist in Invariant Explorer, it will be created before adding traces.

### **🔹 Gemini Integration**

Gateway supports the Gemini `generateContent` and `streamGenerateContent` methods.

1. Follow [these steps](https://ai.google.dev/gemini-api/docs/api-key) to obtain a Gemini API key.
2. **Modify Gemini Client Setup**

   ```python
   import os

   from google import genai

   client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={
            "base_url": "https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/gemini",
            "headers": {
                "Invariant-Authorization": "Bearer your-invariant-api-key"
            },
        },
    )
   ```

   > **Note:** Do not include the curly braces `{}`. If the dataset does not exist in Invariant Explorer, it will be created before adding traces.

### **🔹 OpenAI Swarm Integration**

Integrating directly with a specific agent framework is also supported, simply by configuring the underlying LLM client.

For instance, [OpenAI Swarm](https://github.com/openai/swarm) relies on OpenAI's Python client, the setup is very similar to the standard OpenAI integration:

```python
from swarm import Swarm, Agent
from openai import OpenAI
from httpx import Client
import os

client = Swarm(
    client=OpenAI(
        http_client=Client(headers={"Invariant-Authorization": "Bearer " + os.getenv("INVARIANT_API_KEY", "")}),
        base_url="https://explorer.invariantlabs.ai/api/v1/gateway/weather-swarm-agent/openai",
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
# Output: "It seems to be sunny."
```

### **🔹 LiteLLM Integration**

LiteLLM is a python library that acts as a unified interface for calling multiple LLM providers. If you are using it, it is very convinient to connect to Gateway proxy. You just need to pass the correct `base_url`.

```python
from litellm import completion
import random
import os

base_url = "/api/v1/gateway/litellm/{add-your-dataset-name-here}"
EXAMPLE_MODELS = ["openai/gpt-4o", "gemini/gemini-2.0-flash", "anthropic/claude-3-5-haiku-20241022"]
model = random.choice(SAMPLE_MODELS)

base_url += "/" + model.split("/")[0] # append /gemini /openai or /anthropic. 

if model.split("/")[0] == "gemini":
    base_url += f"/v1beta/models/{model.split('/')[1]}" # gemini expects the model name in the url.


chat_response = completion(
    model=model,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_headers= {"Invariant-Authorization": "Bearer <some-key>"},
    stream=True,
    base_url=base_url,
)

print(chat_response.choices[0].message.content)
# Output: "Paris."
```

### **🔹 Microsoft Autogen Integration**

You can also easily integrate the Gateway with [Microsoft Autogen](https://github.com/microsoft/autogen) as follows:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

import os
from httpx import AsyncClient

async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        http_client=AsyncClient(headers={"Invariant-Authorization": "Bearer " + os.getenv("INVARIANT_API_KEY", "")}),
        base_url="https://explorer.invariantlabs.ai/api/v1/gateway/weather-swarm-agent/openai",
    )
    agent = AssistantAgent("assistant", client)
    print(await agent.run(task="Say 'Hello World!'"))


asyncio.run(main())
# Output: "Hello World!"
```

This will automatically trace your agent interactions in Invariant Explorer.

---

## Quickstart for Users

If you are not building an agent yourself but would like to observe and secure AI agents in your organization, you can do so by configuring the agents to use the Gateway.

See below for example integrations with popular agents.

### **OpenHands Integration**

[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is a platform for software development agents powered by AI.

#### **How to Integrate OpenHands with Invariant Gateway**

##### **Step 1: Modify the API Base**

Enable the `Advanced Options` toggle under Settings and update the `Base URL` to the following

```
https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/openai
```

<img src="./resources/images/openhands-integration.png" height=300/>

##### **Step 2: Adjust the API Key Format**

Set the API Key using the following format:

```text
{your-llm-api-key};invariant-auth={your-invariant-api-key}
```

> **Note:** Do not include the curly braces `{}`.

The Invariant Gateway extracts the `invariant-auth` field from the API key and correctly forwards it to Invariant Explorer while sending the actual API key to OpenAI or Anthropic.

---

### **SWE-agent Integration**

[SWE-agent](https://github.com/SWE-agent/SWE-agent) allows your preferred language model (e.g., GPT-4o or Claude Sonnet 3.5) to autonomously utilize tools for various tasks, such as fixing issues in real GitHub repositories.

#### **Using SWE-agent with Invariant Gateway**

SWE-agent does not support custom headers, so you **cannot** pass the Invariant API Key via `Invariant-Authorization`. However, **there is a workaround** using the Invariant Gateway.

##### **Step 1: Modify the API Base**

Run `sweagent` with the following flag:

```bash
--agent.model.api_base=https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/openai
```

> **Note:** Do not include the curly braces `{}`.

##### **Step 2: Adjust the API Key Format**

Instead of setting your API Key normally, modify the environment variable as follows:

```bash
export OPENAI_API_KEY={your-openai-api-key};invariant-auth={your-invariant-api-key}
export ANTHROPIC_API_KEY={your-anthropic-api-key};invariant-auth={your-invariant-api-key}
```

> **Note:** Do not include the curly braces `{}`.

This setup ensures that SWE-agent works seamlessly with Invariant Gateway, maintaining compatibility while enabling full functionality. 🚀

### **Using MCP with Invariant Gateway**
Invariant Gateway supports MCP (stdio, SSE and Streamable HTTP) tool calling.

For stdio transport based MCP, follow steps [here](https://github.com/invariantlabs-ai/invariant-gateway/tree/main/gateway/mcp).

For **SSE transport based MCP**, here are the steps to point your MCP client to a local instance of the Invariant Gateway which will then proxy all calls to the MCP server while guardrailing:

* Run the Gateway locally by following the steps [here](https://github.com/invariantlabs-ai/invariant-gateway/tree/main?tab=readme-ov-file#run-the-gateway-locally).
* Use the following configuration to connect to the local Gateway instance:
```python
from mcp.client.sse import sse_client

await connect_to_sse_server(
            server_url="http://localhost:8005/api/v1/gateway/mcp/sse",
            headers={
                "MCP-SERVER-BASE-URL": "<The base URL to your MCP server>",
                "INVARIANT-PROJECT-NAME": "<The Invariant dataset name>",
                "PUSH-INVARIANT-EXPLORER": "true",
                "INVARIANT-API-KEY": "<your-invariant-api-key>"
                "INVARIANT-X-MCP-SERVER-{CUSTOM-MCP-SERVER-HEADER-NAME}": "<custom-value-passed-to-mcp-server>"
            },
        )
```

For **Streamable HTTP transport based MCP**, here are the steps to point your MCP client to a local instance of the Invariant Gateway which will then proxy all calls to the MCP server while guardrailing:

* Run the Gateway locally by following the steps [here](https://github.com/invariantlabs-ai/invariant-gateway/tree/main?tab=readme-ov-file#run-the-gateway-locally).
* Use the following configuration to connect to the local Gateway instance:
```python
from mcp.client.streamable_http import streamablehttp_client

await streamablehttp_client(
            url="http://localhost:8005/api/v1/gateway/mcp/sse",
            headers={
                "MCP-SERVER-BASE-URL": "<The base URL to your MCP server>",
                "INVARIANT-PROJECT-NAME": "<The Invariant dataset name>",
                "PUSH-INVARIANT-EXPLORER": "true",
                "INVARIANT-API-KEY": "<your-invariant-api-key>"
                "INVARIANT-X-MCP-SERVER-{CUSTOM-MCP-SERVER-HEADER-NAME}": "<custom-value-passed-to-mcp-server>"
            },
        )
```

The `INVARIANT-API-KEY` header is used both for pushing the traces to explorer and for guardrailing.

If no `INVARIANT-PROJECT-NAME` header is specified but `PUSH-INVARIANT-EXPLORER` is set to "true", a new Invariant project will be created and the MCP traces will be pushed there.

If you pass a header called `INVARIANT-X-MCP-SERVER-CUSTOM-API-KEY`, it will be passed as the `CUSTOM-API-KEY` header to the underlying MCP server.

You can also specify blocking or logging guardrails for the project name by visiting the Explorer.

---

## **Run the Gateway Locally**

You can also operate your own instance of the Gateway, to ensure privacy and security.

To run Gateway locally, you have two options:

### 1. Run gateway from the repository

1. Clone this repository.

2. To start the Invariant Gateway, then run the following commands. Note that you need to have Docker installed.

```bash
cd invariant-gateway
bash run.sh build && bash run.sh up
```

This will launch Gateway at [http://localhost:8005/api/v1/gateway/](http://localhost:8005/api/v1/gateway/docs/).

### 2. Run the Gateway using the published Docker image

You can also run the Gateway using the published Docker image. This is a good option if you want to run the Gateway in a cloud environment.

```bash
# pull the latest image
docker pull --platform linux/amd64 ghcr.io/invariantlabs-ai/invariant-gateway/gateway:latest
# run Gateway on localhost:8005
docker run -p 8005:8005 -e PORT=8005 --platform linux/amd64 ghcr.io/invariantlabs-ai/invariant-gateway/gateway:latest
```

This will launch Gateway at [http://localhost:8005/api/v1/gateway/](http://localhost:8005/api/v1/gateway/docs/). This instance will automatically push your traces to `https://explorer.invariantlabs.ai`. 

### **Set Up an Invariant API Key**

1. Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/api/client-setup/) to obtain an API key. This allows the gateway to push traces to [Invariant Explorer](https://explorer.invariantlabs.ai).

---

## **Development**

### **Pushing to Local Explorer**

By default Gateway points to the public Explorer instance at `explorer.invariantlabs.ai`. To point it to your local Explorer instance, modify the `INVARIANT_API_URL` value inside `.env`. Follow instructions in `.env` on how to point to the local instance.

### **Run Unit Tests**

To run the unit tests, execute:

```bash
bash run.sh unit-tests
```

### **Run Integration Tests**

To run the integration tests, execute:

```bash
bash run.sh integration-tests
```

To run a subset of the integration tests, execute:

```bash
bash run.sh integration-tests open_ai/test_chat_with_tool_call.py
```

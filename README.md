<div align="center">
<h1>Invariant Gateway</h1>

<b>LLM proxy to secure and debug what your AI agents are doing.</b>

</div>

Invariant Gateway is a lightweight _zero-configuration_ service that acts as an intermediary between AI Agents and LLM providers (such as OpenAI and Anthropic).

Gateway automatically traces agent interactions and stores them in the [Invariant Explorer](https://explorer.invariantlabs.ai/), giving you insights into what your agents are doing.

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
- [x] **Seamless forwarding and LLM streaming** to OpenAI, Anthropic, and other LLM providers.
- [x] **Store and organize runtime traces** in the Invariant Explorer.

## **Quickstart for Teams and Users**

Looking to observe and secure AI agents in your organization? See our [no-code quickstart guide for users](#quickstart-for-users) to get started.

## **Quickstart for Developers**

To add Gateway to your agentic system, simply follow the integration guides below depending on the LLM provider you are using.

## **Integration Guides**

### **🔹 OpenAI Integration**

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
# Output: "It seems to be sunny."
```

---

## Quickstart for Users

If you are not building an agent yourself but would like to observe and secure AI agents in your organization, you can do so by configuring the agents to use the Gateway.

See below for example integrations with popular agents.

### **OpenHands Integration**

[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is a platform for software development agents powered by AI.

#### **How to Integrate OpenHands with Invariant Gateway**

##### **Step 1: Modify the API Base**

Enable the `Advanced Options` toggle under settings and update the `Base URL` to the following

```
https://explorer.invariantlabs.ai/api/v1/gateway/{add-your-dataset-name-here}/openai
```

<img src="./resources/images/openhands-integration.png" height=300/>

##### **Step 2: Adjust the API Key Format**

Set the API Key using the following format:

```text
{your-llm-api-key}|invariant-auth: {your-invariant-api-key}
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
export OPENAI_API_KEY={your-openai-api-key}|invariant-auth: {your-invariant-api-key}
export ANTHROPIC_API_KEY={your-anthropic-api-key}|invariant-auth: {your-invariant-api-key}
```

> **Note:** Do not include the curly braces `{}`.

This setup ensures that SWE-agent works seamlessly with Invariant Gateway, maintaining compatibility while enabling full functionality. 🚀

---

## **Run the Gateway Locally**

First, clone this repository. To start the Invariant Gateway, run:

```bash
bash run.sh build && bash run.sh up
```

This will launch Gateway at [http://localhost:8005/api/v1/gateway/](http://localhost:8005/api/v1/gateway/docs/).

### **Set Up an Invariant API Key**

1. Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/api/client-setup/) to obtain an API key. This allows the gateway to push traces to [Invariant Explorer](https://explorer.invariantlabs.ai).

---

## **Development**

### **Pushing to Local Explorer**

By default Gateway points to the public Explorer instance at `explorer.invariantlabs.ai`. To point it to your local Explorer instance, modify the `INVARIANT_API_URL` value inside `.env`. Follow instructions in `.env` on how to point to the local instance.

### **Run Tests**

To run tests, execute:

```bash
./run.sh tests
```

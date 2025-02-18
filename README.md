# **Invariant Proxy**

Invariant Proxy is a lightweight Docker service that acts as an intermediary between AI Agents and LLM providers (such as OpenAI and Anthropic). It captures and forwards agent interactions to the [Invariant Explorer](https://explorer.invariantlabs.ai/), enabling seamless debugging, visualization, and exploration of traces.

![Invariant Proxy Diagram](resources/images/invariant-proxy.png)

---

## **Why Use Invariant Proxy?**
- ✅ **Intercept AI interactions** for better debugging and analysis.
- ✅ **Seamlessly forward API requests** to OpenAI, Anthropic, and other LLM providers (**supports streaming responses too**).
- ✅ **Works with AI Agent platforms and systems** like OpenHands, SWE-agent, etc.
- ✅ **Automatically store and organize traces** in the Invariant Explorer.

---

## **Getting Started**

### **Run the Proxy Locally**
To start the Invariant Proxy, run:

```bash
bash run.sh build && bash run.sh up
```

This will launch the proxy at [http://localhost:8005/api/v1/proxy/](http://localhost:8005/api/v1/proxy/docs/).

### **Set Up an Invariant API Key**
1. Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/Explorer_API/1_client_setup/) to obtain an API key. This allows the proxy to push traces to [Invariant Explorer](https://explorer.invariantlabs.ai).

---

## **Integration Guides**

### **🔹 OpenAI Integration**

1. Follow [these steps](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key) to obtain an OpenAI API key.
2. **Modify OpenAI Client Setup**

   Instead of connecting directly to OpenAI, configure your `OpenAI` client to use the proxy:

   ```python
   from httpx import Client
   from openai import OpenAI

   client = OpenAI(
       http_client=Client(
           headers={
               "Invariant-Authorization": "Bearer your-invariant-api-key"
           },
       ),
       base_url="http://localhost:8005/api/v1/proxy/{add-your-dataset-name-here}/openai",
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
       base_url="http://localhost:8005/api/v1/proxy/{add-your-dataset-name-here}/anthropic",
   )
   ```

   > **Note:** Do not include the curly braces `{}`. If the dataset does not exist in Invariant Explorer, it will be created before adding traces.
---

## **OpenHands Integration**
[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is a platform for software development agents powered by AI.

### **How to Integrate OpenHands with Invariant Proxy**

#### **Step 1: Modify the API Base**
Enable the `Advanced Options` toggle under settings and update the `Base URL`:

<img src="./resources/images/openhands-integration.png" height=300/>

#### **Step 2: Adjust the API Key Format**
Set the API Key using the following format:

```text
{your-llm-api-key}|invariant-auth: {your-invariant-api-key}
```

> **Note:** Do not include the curly braces `{}`.

The Invariant Proxy extracts the `invariant-auth` field from the API key and correctly forwards it to Invariant Explorer while sending the actual API key to OpenAI or Anthropic.

---

## **SWE-agent Integration**
[SWE-agent](https://github.com/SWE-agent/SWE-agent) allows your preferred language model (e.g., GPT-4o or Claude Sonnet 3.5) to autonomously utilize tools for various tasks, such as fixing issues in real GitHub repositories.

### **Using SWE-agent with Invariant Proxy**
SWE-agent does not support custom headers, so you **cannot** pass the Invariant API Key via `Invariant-Authorization`. However, **there is a workaround** using the Invariant Proxy.

#### **Step 1: Modify the API Base**

Run `sweagent` with the following flag:

```bash
--agent.model.api_base=http://localhost:8005/api/v1/proxy/{add-your-dataset-name-here}/openai
```

> **Note:** Do not include the curly braces `{}`.

#### **Step 2: Adjust the API Key Format**
Instead of setting your API Key normally, modify the environment variable as follows:

```bash
export OPENAI_API_KEY={your-openai-api-key}|invariant-auth: {your-invariant-api-key}
export ANTHROPIC_API_KEY={your-anthropic-api-key}|invariant-auth: {your-invariant-api-key}
```

> **Note:** Do not include the curly braces `{}`.

This setup ensures that SWE-agent works seamlessly with Invariant Proxy, maintaining compatibility while enabling full functionality. 🚀

---

## **Development**

### **Run Tests**
To run tests, execute:

```bash
./run.sh tests
```


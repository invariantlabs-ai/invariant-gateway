# **Invariant Proxy**

Invariant Proxy is a lightweight Docker service that acts as an intermediary between AI Agents and LLM providers (such as OpenAI and Anthropic). It captures and forwards agent interactions to the [Invariant Explorer](https://explorer.invariantlabs.ai/), enabling seamless debugging, visualization and exploration of traces.

![Invariant Proxy Diagram](resources/images/invariant-proxy.png)

## **Why Use Invariant Proxy?**
- ✅ **Intercept AI interactions** for better debugging and analysis.
- ✅ **Seamlessly forward API requests** to OpenAI, Anthropic, and other LLM providers (**supports streaming responses too**).
- ✅ **Automatically store and organize traces** in the Invariant Explorer.

## **Getting Started**

**Run the proxy locally**
```bash
bash run.sh build && bash run.sh up
```

This will start the Invariant Proxy on [http://localhost:8005/api/v1/proxy/](http://localhost:8005/api/v1/proxy/docs/).

To integrate the Proxy with your AI agent, you’ll need to modify how your client interacts with LLM providers.

**Setup an Invariant API Key**
1. Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/Explorer_API/1_client_setup/) to obtain an API key. This will allow the Proxy to push traces to [https://explorer.invariantlabs.ai](https://explorer.invariantlabs.ai).

**🔹 OpenAI Integration**

2. To setup an OpenAI API key follow the steps [here](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key).

3. **Modify OpenAI Client Setup**  
   Instead of connecting directly to OpenAI, configure your `OpenAI` client to use the proxy.

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
   # If a dataset with the given name already doesn't exist in Invariant Explorer, this will create the dataset
   # before adding the traces to it.
   # Make API requests to OpenAI using the client as usual.

**🔹 Anthropic Integration**

2. To setup an Anthropic API key follow the steps [here](https://docs.anthropic.com/en/docs/initial-setup#set-your-api-key).

3. **Modify Anthropic Client Setup**  
   Instead of connecting directly to Anthropic, configure your `Anthropic` client to use the proxy.

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
   
   # If a dataset with the given name already doesn't exist in Invariant Explorer, this will create the dataset
   # before adding the traces to it.
   # Make API requests to Anthropic using the client as usual.

## **OpenHands Integration**
[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is a platform for software development agents powered by AI.

OpenHands does not support custom headers, meaning you **cannot** pass the Invariant API Key via the `Invariant-Authorization` header directly.  However, **there is a workaround** using the Invariant Proxy.

#### **Step 1: Modify the API Base**

Enable the `Advanced Options` toggle under settings and update the `Base URL` in the modal like:

<img src="./resources/images/openhands-integration.png" height=300/>


#### **Step 2: Adjust the API Key Format**
Instead of setting your OPENAI_API_KEY (or ANTHROPIC_API_KEY) normally in the settings modal under `API Key` you will need to change the format.

Updated format: ```{your-llm-api-key}|invariant-auth: {your-invariant-api-key}``` without the curly braces.

The Invariant Proxy extracts the invariant-auth field from the API key and correctly forwards the Invariant API Key, allowing traces to be pushed to Invariant Explorer. The request is correctly passed to OpenAI (or Anthropic) with the actual API Key.

This setup ensures that OpenHands works seamlessly with Invariant Proxy, maintaining compatibility while enabling full functionality. 🚀

## SWE-agent Integration

[SWE-agent](https://github.com/SWE-agent/SWE-agent) allows your preferred language model (e.g., GPT-4o or Claude Sonnet 3.5) to autonomously utilize tools for various tasks, such as:

- Fixing issues in real GitHub repositories
- Performing tasks on the web
- Cracking cybersecurity challenges
- Executing custom-defined tasks

### Using SWE-agent with Invariant Proxy

SWE-agent does not support custom headers, meaning you **cannot** pass the Invariant API Key via the `Invariant-Authorization` header directly. However, **there is a workaround** using the Invariant Proxy.

#### **Step 1: Modify the API Base**
When running `sweagent run`, add the following flag to route requests through the Invariant Proxy:

```bash
--agent.model.api_base=http://localhost:8005/api/v1/proxy/{add-your-dataset-name-here}/openai
```
without the curly braces.


#### **Step 2: Adjust the API Key Format**
Instead of setting your OPENAI_API_KEY (or ANTHROPIC_API_KEY) normally, modify the environment variable as follows:

```bash
export OPENAI_API_KEY={your-openai-api-key}|invariant-auth: {your-invariant-api-key}
export ANTHROPIC_API_KEY={your-anthropic-api-key}|invariant-auth: {your-invariant-api-key}
```
without the curly braces.

The Invariant Proxy extracts the invariant-auth field from the API key and correctly forwards the Invariant API Key, allowing traces to be pushed to Invariant Explorer. The request is correctly passed to OpenAI (or Anthropic) with the actual API Key.

This setup ensures that SWE-agent works seamlessly with Invariant Proxy, maintaining compatibility while enabling full functionality. 🚀

## Dev 

### Run tests
```bash
./run.sh tests
```

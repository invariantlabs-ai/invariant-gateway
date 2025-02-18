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
1. Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/Explorer_API/1_client_setup/) to obtain an API key. This will allow the Proxy to push traces to [https://explorer.invariantlabs.ai](https://explorer.invariantlabs.ai), 

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

### Run tests
```bash
./run.sh tests
```

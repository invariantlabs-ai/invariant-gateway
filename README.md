# **Explorer Proxy**

Explorer Proxy is a lightweight Docker service that acts as an intermediary between AI Agents and LLM providers (such as OpenAI and Anthropic). It captures and forwards agent interactions to the [Invariant Explorer](https://explorer.invariantlabs.ai/), enabling seamless visualization and exploration of traces.

## **Why Use Explorer Proxy?**
- ✅ **Intercept AI interactions** for better debugging and analysis.
- ✅ **Seamlessly forward API requests** to OpenAI, Anthropic, and other LLM providers.
- ✅ **Automatically store and organize traces** in Invariant Explorer.

## **Getting Started**
To integrate Explorer Proxy with your AI agent, you’ll need to modify how your client interacts with LLM providers.

### **🔹 OpenAI Integration**
1. **Get an API Key**  
   Follow the instructions [here](https://explorer.invariantlabs.ai/docs/explorer/Explorer_API/1_client_setup/) to obtain an API key.

2. **Modify OpenAI Client Setup**  
   Instead of connecting directly to OpenAI, configure your `OpenAI` client to use the proxy.

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

### **🔹 Anthropic Integration**
Coming Soon!

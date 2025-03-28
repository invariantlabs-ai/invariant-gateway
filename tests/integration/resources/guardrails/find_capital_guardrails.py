# For Anthropic and OpenAI tests:
raise "Madrid detected in the response" if:
    (msg: Message)
   "Madrid" in msg.content

# For Gemini tests:
raise "Dublin detected in the response" if:
    (msg: Message)
    (chunk: str) in text(msg.content)
    "Dublin" in chunk

# For all provider tests:
raise "get_capital is called with Germany as argument" if:
    (call: ToolCall)
    call is tool:get_capital
    call.function.arguments["country_name"] == "Germany"
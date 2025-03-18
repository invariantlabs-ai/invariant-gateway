raise "Madrid detected in the response" if:
    (msg: Message)
   "Madrid" in msg.content

raise "get_capital is called with Germany as argument" if:
    (call: ToolCall)
    call is tool:get_capital
    call.function.arguments["country_name"] == "Germany"
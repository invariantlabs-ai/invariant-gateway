"""Common constants used in the gateway."""

IGNORED_HEADERS = [
    "accept-encoding",
    "host",
    "invariant-authorization",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
    "x-real-ip",
]

CLIENT_TIMEOUT = 60.0

# MCP related constants
UTF_8 = "utf-8"
MCP_METHOD = "method"
MCP_TOOL_CALL = "tools/call"
MCP_LIST_TOOLS = "tools/list"
MCP_PARAMS = "params"
MCP_RESULT = "result"
MCP_SERVER_INFO = "serverInfo"
MCP_CLIENT_INFO = "clientInfo"
INVARIANT_GUARDRAILS_BLOCKED_MESSAGE = """
                    [Invariant Guardrails] The MCP tool call was blocked for security reasons. 
                    Do not attempt to circumvent this block, rather explain to the user based 
                    on the following output what went wrong: %s
                    """
INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE = """
                    [Invariant Guardrails] This server was blocked from advertising its tools due to a security guardrail failure.
                    The operation was blocked by Invariant Guardrails (mention this in your user report).
                    When users ask about this tool, inform them that it was blocked due to a security guardrail failure.
                    %s
                    """
MCP_SERVER_BASE_URL_HEADER = "mcp-server-base-url"

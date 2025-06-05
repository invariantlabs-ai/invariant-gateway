"""This is a simple example of how to use the MCP client with SSE transport."""

from typing import Any
from datetime import timedelta

from mcp import ClientSession, types
from mcp.client.sse import sse_client


async def run(
    gateway_url: str,
    tool_name: str,
    tool_args: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> types.CallToolResult | types.ListToolsResult:
    """
    Run the MCP client with the given parameters.

    Args:
        gateway_url: URL of the Invariant Gateway
        push_to_explorer: Whether to push traces to the Invariant Explorer
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool call
        headers: Optional headers to include in the request
    """
    client = sse_client(
        url=gateway_url,
        timeout=5,
        headers=headers or {},
        sse_read_timeout=10,
    )
    async with client as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            # list tools
            listed_tools = await session.list_tools()
            # call tool
            if tool_name == "tools/list":
                return listed_tools
            else:
                return await session.call_tool(
                    tool_name, tool_args, read_timeout_seconds=timedelta(seconds=10)
                )

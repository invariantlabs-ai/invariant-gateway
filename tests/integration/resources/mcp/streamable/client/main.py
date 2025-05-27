"""This is a simple example of how to use the MCP client with Streamable HTTP transport."""

import asyncio
from datetime import timedelta

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def run(
    gateway_url: str,
    push_to_explorer: bool,
    tool_name: str,
    tool_args: dict[str, Any],
    headers: dict[str, str] = None,
):
    """
    Run the MCP client with the given parameters.

    Args:
        gateway_url: URL of the Invariant Gateway
        push_to_explorer: Whether to push traces to the Invariant Explorer
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool call

    """
    try:
        streams_context = streamablehttp_client(
            url=gateway_url,
            headers=headers or {},
            timeout=timedelta(seconds=5),
            sse_read_timeout=timedelta(seconds=10),
        )
        async with streams_context as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
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
    finally:
        # Sleep for a while to allow the server to process the background tasks
        # like pushing traces to the explorer
        if push_to_explorer:
            await asyncio.sleep(2)

"""This is a simple example of how to use the MCP client with Streamable HTTP transport."""

# pylint: disable=E1101
# pylint: disable=W0201
# pylint: disable=C2801

import asyncio
from datetime import timedelta

from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPClient:
    """MCP Client for interacting with a MCP Streamable HTTP server and processing queries"""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None  # Initialize these to None
        self._session_context = None  # so they always exist

    async def connect_to_streamable_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """
        Connect to an MCP server running with Streamable HTTP transport

        Args:
            server_url: URL of the MCP server
            headers: Optional headers to include in the request
        """
        # Store the context managers so they stay alive
        self._streams_context = streamablehttp_client(
            url=server_url,
            headers=headers or {},
            timeout=timedelta(seconds=5),
            sse_read_timeout=timedelta(seconds=10),
        )
        read_stream, write_stream, session_id = await self._streams_context.__aenter__()
        self.session_id = session_id

        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, tool_name: str, tool_args: dict) -> str:
        """Process a query using MCP server"""
        result = await self.session.call_tool(
            tool_name, tool_args, read_timeout_seconds=timedelta(seconds=10)
        )
        return result


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
    client = MCPClient()
    try:
        await client.connect_to_streamable_server(
            server_url=gateway_url, headers=headers or {}
        )
        # list tools
        listed_tools = await client.session.list_tools()
        # call tool
        if tool_name == "tools/list":
            return listed_tools
        else:
            return await client.process_query(tool_name, tool_args)
    finally:
        # Sleep for a while to allow the server to process the background tasks
        # like pushing traces to the explorer
        if push_to_explorer:
            await asyncio.sleep(2)
        await client.cleanup()

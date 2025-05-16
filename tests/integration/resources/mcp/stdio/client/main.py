"""A MCP client implementation that interacts with MCP server to make tool calls."""

import os

from datetime import timedelta
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP Client for interacting with a MCP stdio server and processing queries"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(
        self,
        invariant_gateway_package_whl_file: str,
        project_name: str,
        server_script_path: str,
        push_to_explorer: bool,
        metadata_keys: Optional[dict[str, str]] = None,
    ):
        """
        Connect to an MCP server.

        Args:
            invariant_gateway_package_whl_file: Path to the Invariant Gateway package
                .whl file
            project_name: Name of the project in Invariant Explorer
            server_script_path: Path to the server script
            push_to_explorer: Whether to push traces to the Invariant Explorer
        """
        args = [
            "--from",
            invariant_gateway_package_whl_file,
            "invariant-gateway",
            "mcp",
            "--project-name",
            project_name,
        ]
        # add metadata cli args
        if metadata_keys is not None:
            for key, value in metadata_keys.items():
                args.append("--metadata-" + key + "=" + value)


        if push_to_explorer:
            args.append("--push-explorer")
        args.extend(
            [
                "--exec",
                "uv",
                "--directory",
                os.path.abspath(os.path.dirname(server_script_path)),
                "run",
                os.path.basename(server_script_path),
            ]
        )

        server_params = StdioServerParameters(
            command="uvx",
            args=args,
            env={
                "INVARIANT_API_KEY": os.environ.get("INVARIANT_API_KEY"),
                "INVARIANT_API_URL": "http://invariant-gateway-test-explorer-app-api:8000",
            },
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.stdio, self.write, read_timeout_seconds=timedelta(seconds=15)
            )
        )

        # initialize the session
        await self.session.initialize()

    async def call_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> types.CallToolResult:
        """
        Make a tool call on the MCP server.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool call
        """
        # Execute tool call
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run(
    invariant_gateway_package_whl_file: str,
    project_name: str,
    server_script_path: str,
    push_to_explorer: bool,
    tool_name: str,
    tool_args: dict[str, Any],
    metadata_keys: Optional[dict[str, str]] = None,
) -> types.CallToolResult:
    """
    Main function to setup the MCP client and server.
    It calls a tool on the server with the given args.

    Args:
        invariant_gateway_package_whl_file: Path to the Invariant Gateway package
            .whl file
        project_name: Name of the project in Invariant Explorer
        server_script_path: Path to the server script
        push_to_explorer: Whether to push traces to the Invariant Explorer
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool call
    """

    client = MCPClient()
    try:
        await client.connect_to_server(
            invariant_gateway_package_whl_file,
            project_name,
            server_script_path,
            push_to_explorer,
            metadata_keys=metadata_keys
        )
        listed_tools = await client.session.list_tools()
        if tool_name == "tools/list":
            # list tools
            return listed_tools
        else:
            return await client.call_tool(tool_name, tool_args)
    finally:
        await client.cleanup()

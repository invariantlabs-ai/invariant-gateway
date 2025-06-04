"""A MCP client implementation that interacts with MCP server to make tool calls."""

import os

from datetime import timedelta
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


def _get_server_params(
    invariant_gateway_package_whl_file: str,
    project_name: str,
    server_script_path: str,
    push_to_explorer: bool,
    metadata_keys: Optional[dict[str, str]] = None,
) -> StdioServerParameters:
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

    return StdioServerParameters(
        command="uvx",
        args=args,
        env={
            "INVARIANT_API_KEY": os.environ.get("INVARIANT_API_KEY"),
            "INVARIANT_API_URL": os.environ.get("INVARIANT_API_URL"),
        },
    )


async def run(
    invariant_gateway_package_whl_file: str,
    project_name: str,
    server_script_path: str,
    push_to_explorer: bool,
    tool_name: str,
    tool_args: dict[str, Any],
    metadata_keys: Optional[dict[str, str]] = None,
) -> types.CallToolResult | types.ListToolsResult:
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
        metadata_keys: Optional metadata keys to include in the request
    """
    client = stdio_client(
        _get_server_params(
            invariant_gateway_package_whl_file,
            project_name,
            server_script_path,
            push_to_explorer,
            metadata_keys=metadata_keys,
        )
    )
    async with client as (stdio, write):
        async with ClientSession(
            stdio, write, read_timeout_seconds=timedelta(seconds=10)
        ) as session:
            await session.initialize()
            # list tools
            listed_tools = await session.list_tools()
            # call tool
            if tool_name == "tools/list":
                return listed_tools
            else:
                return await session.call_tool(tool_name, tool_args)

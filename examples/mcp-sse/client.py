import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_sse_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(
            url=server_url,
            headers=headers or {},
        )
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, tool_name: str, tool_args: dict) -> str:
        """Process a query using MCP server"""
        result = await self.session.call_tool(tool_name, tool_args)
        return result


async def main():
    client = MCPClient()
    try:
        await client.connect_to_sse_server(
            server_url="http://localhost:8005/api/v1/gateway/mcp/sse",
            headers={
                "MCP-SERVER-BASE-URL": "http://localhost:8123",
                "INVARIANT-PROJECT-NAME": "test-mcp-187eghb",
                "PUSH-INVARIANT-EXPLORER": "true",
            },
        )
        print("Hello world I have crossed connect_to_sse_server")
        try:
            result = await client.process_query(
                tool_name="get_alerts",
                tool_args={"state": "NY"},
            )
            print("Result 1: ", result, flush=True)
        except Exception as e:
            print("Error in processing queryyyyy: ", e, flush=True)
            import traceback
            traceback.print_exc()
        result = await client.process_query(
            tool_name="get_forecast",
            tool_args={"latitude": 47.6062, "longitude": -122.3321},
        )
        print("Result 2: ", result, flush=True)
        # result = await client.process_query(
        #     tool_name="get_alerts",
        #     tool_args={"state": "CA"},
        # )
        # print("Result 2: ", result, flush=True)
        # result = await client.process_query(
        #     tool_name="get_alerts",
        #     tool_args={"state": "AZ"},
        # )
        # print("Result 3: ", result, flush=True)
        # result = await client.process_query(
        #     tool_name="get_alerts",
        #     tool_args={"state": "AR"},
        # )
        # print("Result 4: ", result, flush=True)
    except Exception as e:
        print("Error in main: ", e, flush=True)
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

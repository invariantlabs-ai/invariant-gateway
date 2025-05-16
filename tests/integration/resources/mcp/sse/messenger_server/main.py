"""This is a messenger server implementation that returns a few messages based on the username."""

import argparse
import hashlib

import uvicorn

from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# Initialize FastMCP server
mcp = FastMCP("messenger_server")


MESSAGES = [
    "What about you?",
    "What are you doing?",
    "What is your name?",
    "What is your favorite color?",
    "What is your favorite food?",
    "What is your favorite movie?",
    "What is your favorite book?",
]


def _deterministic_index_from_username(username: str, limit: int) -> int:
    """Deterministically calculate the index of messages to return based on the username."""
    hash_val = int(hashlib.sha256(username.encode()).hexdigest(), 16)
    return hash_val % limit + 1


@mcp.tool()
async def get_last_message_from_user(username: str) -> str:
    """Get the last message sent by the username."""
    return MESSAGES[_deterministic_index_from_username(username, len(MESSAGES))] + "\n"


@mcp.tool()
async def send_message(username: str, message: str) -> str:
    """Send a message to the username."""
    return f"Message '{message}' sent to {username}."


def create_starlette_app(server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # pylint: disable=W0212
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # pylint: disable=W0212

    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument("--host", help="Host to bind to", required=True)
    parser.add_argument("--port", help="Port to listen on", required=True, type=int)
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)

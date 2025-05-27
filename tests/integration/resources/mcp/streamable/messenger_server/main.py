"""This is a messenger server implementation that returns a few messages based on the username."""

import argparse
import hashlib
import os

import uvicorn

from mcp.server.fastmcp import FastMCP

# Read config from environment variables
TRANSPORT = os.getenv("TRANSPORT", "json").lower()
STATEFUL = os.getenv("STATEFUL", "false").lower() == "true"

# Initialize FastMCP server
mcp = FastMCP(
    "messenger_server",
    json_response=(TRANSPORT == "json"),
    stateless_http=(not STATEFUL),
)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP Streamable HTTP based server")
    parser.add_argument("--host", help="Host to bind to", required=True)
    parser.add_argument("--port", help="Port to listen on", required=True, type=int)
    parser.add_argument(
        "--transport",
        help="Transport type (json or sse)",
        default="json",
        type=str,
    )
    parser.add_argument(
        "--stateful",
        help="Whether the server is stateful or stateless",
        default="false",
        type=str,
    )
    args = parser.parse_args()

    uvicorn.run(mcp.streamable_http_app, host=args.host, port=args.port)

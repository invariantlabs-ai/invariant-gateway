"""Gateway service to forward requests to the MCP Streamable HTTP servers"""

from gateway.common.mcp_utils import get_mcp_server_base_url

from fastapi import APIRouter, Request, Response


MCP_SESSION_ID_HEADER = "mcp-session-id"

gateway = APIRouter()


def get_session_id(request: Request) -> str | None:
    """Extract the session ID from request headers."""
    return request.headers.get(MCP_SESSION_ID_HEADER)


@gateway.post("/mcp/streamable")
async def mcp_post_gateway(
    request: Request,
) -> Response:
    """
    Forward a POST request to the MCP Streamable server.
    """
    mcp_server_base_url = get_mcp_server_base_url(request)
    pass


@gateway.get("/mcp/streamable")
async def mcp_get_gateway(
    request: Request,
) -> Response:
    """
    Forward a GET request to the MCP Streamable server.

    This allows the server to communicate to the client without the client
    first sending data via HTTP POST. The server can send JSON-RPC requests
    and notifications on this stream.
    """
    mcp_server_base_url = get_mcp_server_base_url(request)
    pass


@gateway.delete("/mcp/streamable")
async def mcp_delete_gateway(
    request: Request,
) -> Response:
    """
    Forward a DELETE request to the MCP Streamable server for explicit session termination.
    """
    mcp_server_base_url = get_mcp_server_base_url(request)
    pass

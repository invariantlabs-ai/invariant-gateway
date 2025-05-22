"""MCP utility functions."""

import re

from fastapi import Request, HTTPException

from gateway.common.constants import MCP_SERVER_BASE_URL_HEADER


def _convert_localhost_to_docker_host(mcp_server_base_url: str) -> str:
    """
    Convert localhost or 127.0.0.1 in an address to host.docker.internal

    Args:
        mcp_server_base_url (str): The original server address from the header

    Returns:
        str: Modified server address with localhost references changed to host.docker.internal
    """
    if "localhost" in mcp_server_base_url or "127.0.0.1" in mcp_server_base_url:
        # Replace localhost or 127.0.0.1 with host.docker.internal
        modified_address = re.sub(
            r"(https?://)(?:localhost|127\.0\.0\.1)(\b|:)",
            r"\1host.docker.internal\2",
            mcp_server_base_url,
        )
        return modified_address

    return mcp_server_base_url


def get_mcp_server_base_url(request: Request) -> str:
    """
    Extract the MCP server base URL from the request headers.

    Args:
        request (Request): The incoming request object.

    Returns:
        str: The MCP server base URL.

    Raises:
        HTTPException: If the MCP server base URL is not found in the headers.
    """
    mcp_server_base_url = request.headers.get(MCP_SERVER_BASE_URL_HEADER)
    if not mcp_server_base_url:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {MCP_SERVER_BASE_URL_HEADER} header",
        )
    return _convert_localhost_to_docker_host(mcp_server_base_url)

"""MCP utility functions."""

import asyncio
import re

from typing import Tuple

from fastapi import Request, HTTPException
from gateway.common.constants import (
    INVARIANT_GUARDRAILS_BLOCKED_MESSAGE,
    INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE,
    MCP_SERVER_BASE_URL_HEADER,
    MCP_PARAMS,
    MCP_RESULT,
)
from gateway.common.guardrails import GuardrailAction
from gateway.common.mcp_sessions_manager import (
    McpSessionsManager,
)
from gateway.integrations.explorer import create_annotations_from_guardrails_errors
from gateway.mcp.log import format_errors_in_response


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
    return _convert_localhost_to_docker_host(mcp_server_base_url).rstrip("/")


async def hook_tool_call(
    session_id: str, session_store: McpSessionsManager, request_body: dict
) -> Tuple[dict, bool]:
    """
    Hook to process the request JSON before sending it to the MCP server.

    Args:
        session_id (str): The session ID associated with the request.
        request_body (dict): The request JSON to be processed.
    """
    tool_call = {
        "id": f"call_{request_body.get('id')}",
        "type": "function",
        "function": {
            "name": request_body.get(MCP_PARAMS).get("name"),
            "arguments": request_body.get(MCP_PARAMS).get("arguments"),
        },
    }
    message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    # Check for blocking guardrails - this blocks until completion
    session = session_store.get_session(session_id)
    guardrails_result = await session.get_guardrails_check_result(
        message, action=GuardrailAction.BLOCK
    )
    # If the request is blocked, return a message indicating the block reason.
    # If there are new errors, run append_and_push_trace in background.
    # If there are no new errors, just return the original request.
    if (
        guardrails_result
        and guardrails_result.get("errors", [])
        and check_if_new_errors(session_id, session_store, guardrails_result)
    ):
        # Add the trace to the explorer
        asyncio.create_task(
            session_store.add_message_to_session(
                session_id=session_id,
                message=message,
                guardrails_result=guardrails_result,
            )
        )
        return {
            "jsonrpc": "2.0",
            "id": request_body.get("id"),
            "error": {
                "code": -32600,
                "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                % guardrails_result["errors"],
            },
        }, True
    # Push trace to the explorer
    await session_store.add_message_to_session(session_id, message, guardrails_result)
    return request_body, False


def check_if_new_errors(
    session_id: str, session_store: McpSessionsManager, guardrails_result: dict
) -> bool:
    """Checks if there are new errors in the guardrails result."""
    session = session_store.get_session(session_id)
    annotations = create_annotations_from_guardrails_errors(
        guardrails_result.get("errors", [])
    )
    for annotation in annotations:
        if annotation not in session.annotations:
            return True
    return False


async def hook_tool_call_response(
    session_id: str,
    session_store: McpSessionsManager,
    response_json: dict,
    is_tools_list=False,
) -> dict:
    """

    Hook to process the response JSON after receiving it from the MCP server.
    Args:
        session_id (str): The session ID associated with the request.
        response_json (dict): The response JSON to be processed.
    Returns:
        dict: The response JSON is returned if no guardrail is violated
              else an error dict is returned.
    """
    blocked = False
    message = {
        "role": "tool",
        "tool_call_id": f"call_{response_json.get('id')}",
        "content": response_json.get(MCP_RESULT).get("content"),
        "error": response_json.get(MCP_RESULT).get("error"),
    }
    result = response_json
    session = session_store.get_session(session_id)
    guardrails_result = await session.get_guardrails_check_result(
        message, action=GuardrailAction.BLOCK
    )

    if (
        guardrails_result
        and guardrails_result.get("errors", [])
        and check_if_new_errors(session_id, session_store, guardrails_result)
    ):
        blocked = True
        # If the request is blocked, return a message indicating the block reason
        if not is_tools_list:
            result = {
                "jsonrpc": "2.0",
                "id": response_json.get("id"),
                "error": {
                    "code": -32600,
                    "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                    % guardrails_result["errors"],
                },
            }
        else:
            # special error response for tools/list tool call
            result = {
                "jsonrpc": "2.0",
                "id": response_json.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "blocked_" + tool["name"],
                            "description": INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE
                            % format_errors_in_response(guardrails_result["errors"]),
                            "inputSchema": {
                                "properties": {},
                                "required": [],
                                "title": "invariant_mcp_server_blockedArguments",
                                "type": "object",
                            },
                            "annotations": {
                                "title": "This tool was blocked by security guardrails.",
                            },
                        }
                        for tool in response_json["result"]["tools"]
                    ]
                },
            }

    # Push trace to the explorer - don't block on its response
    asyncio.create_task(
        session_store.add_message_to_session(session_id, message, guardrails_result)
    )
    return result, blocked

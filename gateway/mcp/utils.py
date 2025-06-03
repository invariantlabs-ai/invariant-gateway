"""MCP utility functions - Updated to work with transport strategy pattern."""

import asyncio
import json
import re
import uuid
from typing import Tuple

from fastapi import Request, HTTPException

from gateway.common.guardrails import GuardrailAction
from gateway.integrations.explorer import create_annotations_from_guardrails_errors
from gateway.mcp.constants import (
    INVARIANT_GUARDRAILS_BLOCKED_MESSAGE,
    INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE,
    INVARIANT_SESSION_ID_PREFIX,
    MCP_CLIENT_INFO,
    MCP_LIST_TOOLS,
    MCP_METHOD,
    MCP_PARAMS,
    MCP_RESULT,
    MCP_SERVER_BASE_URL_HEADER,
    MCP_SERVER_INFO,
    MCP_TOOL_CALL,
)
from gateway.mcp.log import format_errors_in_response
from gateway.mcp.mcp_sessions_manager import McpSession, McpSessionsManager


def generate_session_id() -> str:
    """Generate a new session ID."""
    return INVARIANT_SESSION_ID_PREFIX + uuid.uuid4().hex


def update_mcp_server_in_session_metadata(
    session: McpSession, response_body: dict
) -> None:
    """Update the MCP server information in the session metadata."""
    if response_body.get(MCP_RESULT) and response_body.get(MCP_RESULT).get(
        MCP_SERVER_INFO
    ):
        session.attributes.metadata["mcp_server"] = (
            response_body.get(MCP_RESULT).get(MCP_SERVER_INFO).get("name", "")
        )


def update_tool_call_id_in_session(session: McpSession, request_body: dict) -> None:
    """Updates the tool call ID in the session."""
    if request_body.get(MCP_METHOD) and request_body.get("id"):
        session.id_to_method_mapping[request_body.get("id")] = request_body.get(
            MCP_METHOD
        )


def update_mcp_client_info_in_session(session: McpSession, request_body: dict) -> None:
    """Update the MCP client info in the session metadata."""
    if request_body.get(MCP_PARAMS) and request_body.get(MCP_PARAMS).get(
        MCP_CLIENT_INFO
    ):
        session.attributes.metadata["mcp_client"] = (
            request_body.get(MCP_PARAMS).get(MCP_CLIENT_INFO).get("name", "")
        )


def update_session_from_request(session: McpSession, request_body: dict) -> None:
    """Update the MCP client information and request id in the session."""
    update_mcp_client_info_in_session(session, request_body)
    update_tool_call_id_in_session(session, request_body)


def get_mcp_server_base_url(request: Request) -> str:
    """Extract the MCP server base URL from the request headers."""
    mcp_server_base_url = request.headers.get(MCP_SERVER_BASE_URL_HEADER)
    if not mcp_server_base_url:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {MCP_SERVER_BASE_URL_HEADER} header",
        )
    return _convert_localhost_to_docker_host(mcp_server_base_url).rstrip("/")


def _convert_localhost_to_docker_host(mcp_server_base_url: str) -> str:
    """Convert localhost or 127.0.0.1 in an address to host.docker.internal."""
    if "localhost" in mcp_server_base_url or "127.0.0.1" in mcp_server_base_url:
        modified_address = re.sub(
            r"(https?://)(?:localhost|127\.0\.0\.1)(\b|:)",
            r"\1host.docker.internal\2",
            mcp_server_base_url,
        )
        return modified_address
    return mcp_server_base_url


def _check_if_new_errors(
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


async def hook_tool_call(
    session_id: str, session_store: McpSessionsManager, request_body: dict
) -> Tuple[dict, bool]:
    """
    Hook to process the request JSON before sending it to the MCP server.

    Args:
        session_id (str): The session ID associated with the request.
        session_store (McpSessionsManager): The session store to manage sessions.
        request_body (dict): The request JSON to be processed.

    Returns:
        Tuple[dict, bool]: A tuple hook tool call response as a dict and a boolean
        indicating whether the request was blocked. If the request is blocked, the
        dict will contain an error message else it will contain the original request.
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

    # Check for blocking guardrails
    session = session_store.get_session(session_id)
    guardrails_result = await session.get_guardrails_check_result(
        message, action=GuardrailAction.BLOCK
    )

    # If the request is blocked, return error message
    if (
        guardrails_result
        and guardrails_result.get("errors", [])
        and _check_if_new_errors(session_id, session_store, guardrails_result)
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


async def hook_tool_call_response(
    session_id: str,
    session_store: McpSessionsManager,
    response_body: dict,
    is_tools_list=False,
) -> Tuple[dict, bool]:
    """

    Hook to process the response JSON after receiving it from the MCP server.
    Args:
        session_id (str): The session ID associated with the request.
        session_store (McpSessionsManager): The session store to manage sessions.
        response_body (dict): The response JSON to be processed.
        is_tools_list (bool): Flag to indicate if the response is from a tools/list call.
    Returns:
        Tuple[dict, bool]: A tuple containing the processed response JSON
        and a boolean indicating whether the response was blocked. If the response
        is blocked, the dict will contain an error message else it will contain the
        original response.
    """
    is_blocked = False
    result = response_body

    message = {
        "role": "tool",
        "tool_call_id": f"call_{result.get('id')}",
        "content": result.get(MCP_RESULT, {}).get("content"),
        "error": result.get(MCP_RESULT, {}).get("error"),
    }

    session = session_store.get_session(session_id)
    guardrails_result = await session.get_guardrails_check_result(
        message, action=GuardrailAction.BLOCK
    )

    if (
        guardrails_result
        and guardrails_result.get("errors", [])
        and _check_if_new_errors(session_id, session_store, guardrails_result)
    ):
        is_blocked = True

        if not is_tools_list:
            result = {
                "jsonrpc": "2.0",
                "id": response_body.get("id"),
                "error": {
                    "code": -32600,
                    "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                    % guardrails_result["errors"],
                },
            }
        else:
            # Special error response for tools/list
            result = {
                "jsonrpc": "2.0",
                "id": response_body.get("id"),
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
                        for tool in response_body.get("result", {}).get("tools", [])
                    ]
                },
            }

    # Push trace to the explorer
    await session_store.add_message_to_session(session_id, message, guardrails_result)
    return result, is_blocked


async def intercept_response(
    session_id: str, session_store: McpSessionsManager, response_body: dict
) -> Tuple[dict, bool]:
    """
    Intercept the response and check for guardrails.
    This function is used to intercept responses and check for guardrails.
    If the response is blocked, it returns a message indicating the block
    reason with a boolean flag set to True. If the response is not blocked,
    it returns the original response with a boolean flag set to False.

    Args:
        session_id (str): The session ID associated with the request.
        session_store (McpSessionsManager): The session store to manage sessions.
        response_body (dict): The response JSON to be processed.

    Returns:
        Tuple[dict, bool]: A tuple containing the processed response JSON
        and a boolean indicating whether the response was blocked.
    """
    session = session_store.get_session(session_id)
    method = session.id_to_method_mapping.get(response_body.get("id"))

    intercept_response_result = response_body
    is_blocked = False

    # Intercept and potentially block tool call response
    if method == MCP_TOOL_CALL:
        intercept_response_result, is_blocked = await hook_tool_call_response(
            session_id=session_id,
            session_store=session_store,
            response_body=response_body,
        )
    # Intercept and potentially block list tool call response
    elif method == MCP_LIST_TOOLS:
        # Store tools in metadata
        tools = response_body.get(MCP_RESULT, {}).get("tools", [])
        session_store.get_session(session_id).attributes.metadata["tools"] = tools

        intercept_response_result, is_blocked = await hook_tool_call_response(
            session_id=session_id,
            session_store=session_store,
            response_body={
                "jsonrpc": "2.0",
                "id": response_body.get("id"),
                "result": {
                    "content": json.dumps(tools),
                    "tools": tools,
                },
            },
            is_tools_list=True,
        )

    return intercept_response_result, is_blocked

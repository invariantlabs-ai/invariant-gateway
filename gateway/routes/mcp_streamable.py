"""Gateway service to forward requests to the MCP Streamable HTTP servers"""

import json
import uuid

from typing import Tuple

import httpx

from httpx_sse import aconnect_sse
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from gateway.common.constants import (
    CLIENT_TIMEOUT,
    INVARIANT_SESSION_ID_PREFIX,
    MCP_CLIENT_INFO,
    MCP_LIST_TOOLS,
    MCP_METHOD,
    MCP_PARAMS,
    MCP_RESULT,
    MCP_SERVER_INFO,
    MCP_TOOL_CALL,
    UTF_8,
)
from gateway.common.mcp_sessions_manager import (
    McpSessionsManager,
    McpAttributes,
)
from gateway.common.mcp_utils import (
    get_mcp_server_base_url,
    hook_tool_call,
    hook_tool_call_response,
)

gateway = APIRouter()
session_store = McpSessionsManager()

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"
CONTENT_TYPE_HEADER = "content-type"
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_SERVER_POST_DELETE_HEADERS = {
    "connection",
    "accept",
    "content-length",
    CONTENT_TYPE_HEADER,
    MCP_SESSION_ID_HEADER,
}
MCP_SERVER_GET_HEADERS = {
    "connection",
    "accept",
    "cache-control",
    MCP_SESSION_ID_HEADER,
}


@gateway.post("/mcp/streamable")
async def mcp_post_streamable_gateway(request: Request) -> StreamingResponse:
    """
    Forward a POST request to the MCP Streamable server.
    """
    request_body_bytes = await request.body()
    request_body = json.loads(request_body_bytes)
    sse_header_attributes = McpAttributes.from_request_headers(request.headers)
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    is_initialization_request = _is_initialization_request(request_body)

    if session_id:
        # If a session ID is provided in the request headers, it was already initialized
        # in McpSessionsManager. This might be a session ID returned by the MCP server
        # or a session ID generated in the gateway.
        _update_tool_call_id_in_session(session_id, request_body)
    elif is_initialization_request:
        # If this is an initialization request, we generate a session ID,
        # We don't call initialize_session here because we don't know
        # if the MCP server is running with stateless_http set to True or False.
        # If later in the response from MCP server, we don't receive a session ID then this
        # will be initialized and returned back to the client else this will be
        # overwritten by the session ID returned by the MCP server.
        session_id = _generate_session_id()

    # Intercept the request and check for guardrails.
    if not is_initialization_request:
        request_interception_result = await _intercept_request(session_id, request_body)
        if request_interception_result:
            return request_interception_result

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        try:
            response = await client.post(
                url=_get_mcp_server_endpoint(request),
                headers=_get_headers_for_mcp_post_and_delete(request),
                content=request_body_bytes,
                follow_redirects=True,
            )

            # Try to extract session ID from MCP server response
            resp_session_id = response.headers.get(MCP_SESSION_ID_HEADER)

            # If MCP returned a session ID and we haven't seen, initialize it
            if resp_session_id:
                if not session_store.session_exists(resp_session_id):
                    await session_store.initialize_session(
                        resp_session_id, sse_header_attributes
                    )
                session_id = resp_session_id

            # If no session ID is returned, and this is an init request, initialize our own
            elif is_initialization_request and not session_store.session_exists(
                session_id
            ):
                await session_store.initialize_session(
                    session_id, sse_header_attributes
                )

            # Update client info if this is an initialization request
            if is_initialization_request:
                _update_mcp_client_info_in_session(
                    session_id=session_id,
                    request_body=request_body,
                )

            # If the response is JSON type, handle it as a JSON response.
            if response.headers.get(CONTENT_TYPE_HEADER) == CONTENT_TYPE_JSON:
                return await _handle_mcp_json_response(
                    session_id=session_id,
                    is_initialization_request=is_initialization_request,
                    response=response,
                )

            # Else return SSE streaming response
            return await _handle_mcp_streaming_response(
                session_id=session_id,
                is_initialization_request=is_initialization_request,
                response=response,
            )
        except httpx.RequestError as e:
            print(f"[MCP POST] Request error: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail="Request error") from e
        except Exception as e:
            print(f"[MCP POST] Unexpected error: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail="Unexpected error") from e


@gateway.get("/mcp/streamable")
async def mcp_get_streamable_gateway(
    request: Request,
) -> Response:
    """
    Forward a GET request to the MCP Streamable server.

    This allows the server to communicate to the client without the client
    first sending data via HTTP POST. The server can send JSON-RPC requests
    and notifications on this stream.
    """
    mcp_server_endpoint = _get_mcp_server_endpoint(request)
    response_headers = {}
    filtered_headers = {
        k: v for k, v in request.headers.items() if k.lower() in MCP_SERVER_GET_HEADERS
    }

    async def event_generator():
        """Connect to MCP server and process its events."""

        async with httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT)) as client:
            try:
                async with aconnect_sse(
                    client,
                    "GET",
                    mcp_server_endpoint,
                    headers=filtered_headers,
                ) as event_source:
                    if event_source.response.status_code != 200:
                        error_content = await event_source.response.aread()
                        raise HTTPException(
                            status_code=event_source.response.status_code,
                            detail=error_content,
                        )
                    response_headers.update(dict(event_source.response.headers.items()))

                    async for sse in event_source.aiter_sse():
                        yield sse

            except httpx.StreamClosed as e:
                print(f"Server stream closed: {e}", flush=True)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error processing server events: {e}", flush=True)

    return StreamingResponse(
        event_generator(),
        media_type=CONTENT_TYPE_SSE,
        headers={"X-Proxied-By": "mcp-gateway", **response_headers},
    )


@gateway.delete("/mcp/streamable")
async def mcp_delete_streamable_gateway(
    request: Request,
) -> Response:
    """
    Forward a DELETE request to the MCP Streamable server for explicit session termination.
    """
    session_id = _get_session_id(request)
    if not session_store.session_exists(session_id):
        raise HTTPException(
            status_code=400,
            detail="Session does not exist",
        )
    if session_id.startswith(INVARIANT_SESSION_ID_PREFIX):
        return Response(
            content="",
            status_code=200,
            headers={
                "X-Proxied-By": "mcp-gateway",
            },
        )
    mcp_server_endpoint = _get_mcp_server_endpoint(request)

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        try:
            response = await client.delete(
                url=mcp_server_endpoint,
                headers=_get_headers_for_mcp_post_and_delete(request),
            )
            await session_store.cleanup_session_lock(session_id)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    "X-Proxied-By": "mcp-gateway",
                    **response.headers,
                },
            )

        except httpx.RequestError as e:
            print(f"[MCP DELETE] Request error: {str(e)}")
            raise HTTPException(status_code=500, detail="Request error") from e
        except Exception as e:
            print(f"[MCP DELETE] Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected error") from e


def _get_headers_for_mcp_post_and_delete(
    request: Request,
) -> dict:
    """
    Get headers for MCP server POST and DELETE requests.
    This function filters out headers that are not needed for the MCP server.
    If there is a session ID header, it ensures that it does not start
    with INVARIANT_SESSION_ID_PREFIX since those are generated by the gateway
    and not the MCP server so these should not be sent to the MCP server.
    """
    return {
        k: v
        for k, v in request.headers.items()
        if (
            (k.lower() in MCP_SERVER_POST_DELETE_HEADERS)
            and not (
                k.lower() == MCP_SESSION_ID_HEADER
                and v.startswith(INVARIANT_SESSION_ID_PREFIX)
            )
        )
    }


def _get_session_id(request: Request) -> str:
    """Extract the session ID from request headers."""
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {MCP_SESSION_ID_HEADER} header",
        )
    return session_id


def _get_mcp_server_endpoint(request: Request) -> str:
    """
    Extract the MCP server endpoint from the request headers.
    """
    return get_mcp_server_base_url(request) + "/mcp/"


def _generate_session_id() -> str:
    """
    Generate a new session ID.
    If the MCP server is session less then we don't have a session ID from the MCP server.
    """
    return INVARIANT_SESSION_ID_PREFIX + uuid.uuid4().hex


def _update_tool_call_id_in_session(session_id: str, request_body: dict) -> None:
    """
    Updates the tool call ID in the session.
    """
    session = session_store.get_session(session_id)
    if request_body.get(MCP_METHOD) and request_body.get("id"):
        session.id_to_method_mapping[request_body.get("id")] = request_body.get(
            MCP_METHOD
        )


def _update_mcp_client_info_in_session(session_id: str, request_body: dict) -> None:
    """
    Update the MCP client info in the session metadata.
    """
    session = session_store.get_session(session_id)
    if request_body.get(MCP_PARAMS) and request_body.get(MCP_PARAMS).get(
        MCP_CLIENT_INFO
    ):
        session.attributes.metadata["mcp_client"] = (
            request_body.get(MCP_PARAMS).get(MCP_CLIENT_INFO).get("name", "")
        )


def _update_mcp_response_info_in_session(
    session_id: str, response_json: dict, is_json_response: bool
) -> None:
    """
    Update the MCP response info in the session metadata.
    """
    session = session_store.get_session(session_id)
    if response_json.get(MCP_RESULT) and response_json.get(MCP_RESULT).get(
        MCP_SERVER_INFO
    ):
        session.attributes.metadata["mcp_server"] = (
            response_json.get(MCP_RESULT).get(MCP_SERVER_INFO).get("name", "")
        )
    session.attributes.metadata["server_response_type"] = "json" if is_json_response else "sse"


def _is_initialization_request(request_body: dict) -> bool:
    """
    Check if the request is an initialization request.
    An initialization request is a JSON-RPC request with method "initialize".
    Once initialization is done, the client sends a notification "notifications/initialized".
    This function checks for both cases.
    """
    return (
        request_body.get("method") in ["initialize", "notifications/initialized"]
        and "jsonrpc" in request_body
    )


async def _handle_mcp_json_response(
    session_id: str, is_initialization_request: bool, response: Response
) -> Response:
    """
    Handle the MCP JSON response.
    It checks for guardrails and returns the response accordingly.
    """
    # If the response is blocked by guardrails
    # return the error message else return the response as is
    response_content = response.content
    # The server response is empty string when client sends "notifications/initialized"
    response_json = (
        json.loads(response_content.decode(UTF_8)) if response_content else {}
    )
    if response_json:
        _update_mcp_response_info_in_session(
            session_id=session_id, response_json=response_json, is_json_response=True
        )
    response_code = response.status_code

    if not is_initialization_request:
        intercept_response_result, blocked = await _intercept_response(
            session_id=session_id, response_json=response_json
        )
        if blocked:
            response_content = json.dumps(intercept_response_result).encode(UTF_8)
            response_code = 400

    # Build response headers, injecting gateway generated session ID if missing
    response_headers = {
        "X-Proxied-By": "mcp-gateway",
        **response.headers,
    }
    if MCP_SESSION_ID_HEADER not in response.headers:
        response_headers[MCP_SESSION_ID_HEADER] = session_id
    return Response(
        content=response_content,
        status_code=response_code,
        headers=response_headers,
    )


async def _handle_mcp_streaming_response(
    session_id: str, is_initialization_request: bool, response: Response
) -> StreamingResponse:
    """
    Handle the MCP streaming response.
    It checks for guardrails and returns the response accordingly.
    """

    async def event_generator():
        # Events from MCP server have two parts:
        # 1. event: {type} -> contains the type of event
        # 2. data: {data} -> contains the actual message
        # We are reading line by line so we need to buffer so that we can
        # send the entire event (with both type and data) together.
        # Once we receive an empty line, we end the stream.
        buffer = ""
        async for line in response.aiter_lines():
            stripped_line = line.strip()
            if not stripped_line:
                break  # End of stream
            if buffer:
                response_json = json.loads(stripped_line.split("data: ")[1].strip())
                if not is_initialization_request:
                    (
                        intercept_response_result,
                        blocked,
                    ) = await _intercept_response(
                        session_id=session_id,
                        response_json=response_json,
                    )
                    if blocked:
                        yield (
                            f"{buffer}\n"
                            f"data: {json.dumps(intercept_response_result)}\n\n"
                        )
                        break
                else:
                    _update_mcp_response_info_in_session(
                        session_id=session_id,
                        response_json=response_json,
                        is_json_response=False,
                    )
                yield f"{buffer}\n{stripped_line}\n\n"
                # Clear the buffer for the next event
                buffer = ""
            else:
                buffer = stripped_line

    # Build response headers, injecting gateway generated session ID if missing
    response_headers = {
        "X-Proxied-By": "mcp-gateway",
        **response.headers,
    }
    if MCP_SESSION_ID_HEADER not in response.headers:
        response_headers[MCP_SESSION_ID_HEADER] = session_id

    return StreamingResponse(
        event_generator(),
        media_type=CONTENT_TYPE_SSE,
        headers=response_headers,
    )


async def _intercept_request(session_id: str, request_body: dict) -> Response | None:
    """
    Intercept the request and check for guardrails.
    This function is used to intercept requests and check for guardrails.
    If the request is blocked, it returns a message indicating the block reason.
    """
    if request_body.get(MCP_METHOD) == MCP_TOOL_CALL:
        hook_tool_call_result, is_blocked = await hook_tool_call(
            session_id=session_id,
            session_store=session_store,
            request_body=request_body,
        )
        if is_blocked:
            return Response(
                content=json.dumps(hook_tool_call_result),
                status_code=400,
                media_type="application/json",
            )
    elif request_body.get(MCP_METHOD) == MCP_LIST_TOOLS:
        hook_tool_call_result, is_blocked = await hook_tool_call(
            session_id=session_id,
            session_store=session_store,
            request_body={
                "id": request_body.get("id"),
                "method": MCP_LIST_TOOLS,
                "params": {"name": MCP_LIST_TOOLS, "arguments": {}},
            },
        )
        if is_blocked:
            return Response(
                content=json.dumps(hook_tool_call_result),
                status_code=400,
                media_type="application/json",
            )
    return None


async def _intercept_response(
    session_id: str, response_json: dict
) -> Tuple[dict, bool]:
    """
    Intercept the response and check for guardrails.
    This function is used to intercept responses and check for guardrails.
    If the response is blocked, it returns a message indicating the block
    reason with a boolean flag set to True. If the response is not blocked,
    it returns the original response with a boolean flag set to False.
    """
    session = session_store.get_session(session_id)
    method = session.id_to_method_mapping.get(response_json.get("id"))
    # Intercept and potentially block tool call response
    if method == MCP_TOOL_CALL:
        result, blocked = await hook_tool_call_response(
            session_id=session_id,
            session_store=session_store,
            response_json=response_json,
        )
        return result, blocked
    # Intercept and potentially block list tool call response
    elif method == MCP_LIST_TOOLS:
        # store tools in metadata
        session_store.get_session(session_id).attributes.metadata["tools"] = response_json.get(
            MCP_RESULT
        ).get("tools")
        # store tools/list tool call in trace
        result, blocked = await hook_tool_call_response(
            session_id=session_id,
            session_store=session_store,
            response_json={
                "id": response_json.get("id"),
                "result": {
                    "content": json.dumps(response_json.get(MCP_RESULT).get("tools")),
                    "tools": response_json.get(MCP_RESULT).get("tools"),
                },
            },
            is_tools_list=True,
        )
        return result, blocked
    return response_json, False

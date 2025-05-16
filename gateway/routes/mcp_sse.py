"""Gateway service to forward requests to the MCP SSE servers"""

import asyncio
import json
import re
import os
from typing import Tuple

import httpx
from httpx_sse import aconnect_sse, ServerSentEvent
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from gateway.common.constants import (
    CLIENT_TIMEOUT,
    INVARIANT_GUARDRAILS_BLOCKED_MESSAGE,
    INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE,
    MCP_METHOD,
    MCP_TOOL_CALL,
    MCP_LIST_TOOLS,
    MCP_PARAMS,
    MCP_RESULT,
    MCP_SERVER_INFO,
    MCP_CLIENT_INFO,
)
from gateway.common.guardrails import GuardrailAction
from gateway.common.mcp_sessions_manager import (
    McpSessionsManager,
    SseHeaderAttributes,
)
from gateway.mcp.log import format_errors_in_response
from gateway.integrations.explorer import create_annotations_from_guardrails_errors

MCP_SERVER_POST_HEADERS = {
    "connection",
    "accept",
    "content-length",
    "content-type",
}
MCP_SERVER_SSE_HEADERS = {
    "connection",
    "accept",
    "cache-control",
}

gateway = APIRouter()
session_store = McpSessionsManager()


@gateway.post("/mcp/messages/")
@gateway.post("/mcp/sse/messages/")
async def mcp_post_gateway(
    request: Request,
) -> Response:
    """Proxy calls to the MCP Server tools"""
    query_params = dict(request.query_params)
    if not query_params.get("session_id"):
        return HTTPException(
            status_code=400,
            detail="Missing 'session_id' query parameter",
        )
    if not session_store.session_exists(query_params.get("session_id")):
        return HTTPException(
            status_code=400,
            detail="Session does not exist",
        )
    if not request.headers.get("mcp-server-base-url"):
        return HTTPException(
            status_code=400,
            detail="Missing 'mcp-server-base-url' header",
        )

    session_id = query_params.get("session_id")
    mcp_server_messages_endpoint = (
        _convert_localhost_to_docker_host(request.headers.get("mcp-server-base-url"))
        + "/messages/?"
        + session_id
    )
    request_body_bytes = await request.body()
    request_json = json.loads(request_body_bytes)
    session = session_store.get_session(session_id)
    if request_json.get(MCP_METHOD) and request_json.get("id"):
        session.id_to_method_mapping[request_json.get("id")] = request_json.get(
            MCP_METHOD
        )
    if request_json.get(MCP_PARAMS) and request_json.get(MCP_PARAMS).get(
        MCP_CLIENT_INFO
    ):
        session.metadata["mcp_client_name"] = (
            request_json.get(MCP_PARAMS).get(MCP_CLIENT_INFO).get("name", "")
        )

    if request_json.get(MCP_METHOD) == MCP_TOOL_CALL:
        # Intercept and potentially block the request
        hook_tool_call_result, is_blocked = await _hook_tool_call(
            session_id=session_id, request_json=request_json
        )
        if is_blocked:
            # Add the error message to the session.
            # The error message is sent back to the client using the SSE stream.
            await session.add_pending_error_message(hook_tool_call_result)
            return Response(content="Accepted", status_code=202)
    elif request_json.get(MCP_METHOD) == MCP_LIST_TOOLS:
        # Intercept and potentially block the request
        hook_tool_call_result, is_blocked = await _hook_tool_call(
            session_id=session_id, request_json={
                "id": request_json.get("id"),
                "method": MCP_LIST_TOOLS,
                "params": {
                    "name": MCP_LIST_TOOLS,
                    "arguments": "{}"
                },
            }
        )
        if is_blocked:
            # Add the error message to the session.
            # The error message is sent back to the client using the SSE stream.
            await session.add_pending_error_message(hook_tool_call_result)
            return Response(content="Accepted", status_code=202)

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        try:
            response = await client.post(
                url=mcp_server_messages_endpoint,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() in MCP_SERVER_POST_HEADERS
                },
                json=request_json,
                params=query_params,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    "X-Proxied-By": "mcp-gateway",
                    **response.headers,
                },
            )

        except httpx.RequestError as e:
            print(f"[MCP POST] Request error: {str(e)}")
            raise HTTPException(status_code=500, detail="Request error") from e
        except Exception as e:
            print(f"[MCP POST] Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected error") from e



@gateway.get("/mcp/sse")
async def mcp_get_sse_gateway(
    request: Request,
) -> StreamingResponse:
    """Proxy calls to the MCP Server tools"""
    mcp_server_base_url = request.headers.get("mcp-server-base-url")
    if not mcp_server_base_url:
        print("missing base url", request.headers, flush=True)
        raise HTTPException(
            status_code=400,
            detail="Missing 'mcp-server-base-url' header",
        )
    mcp_server_sse_endpoint = (
        _convert_localhost_to_docker_host(mcp_server_base_url) + "/sse"
    )

    query_params = dict(request.query_params)
    response_headers = {}
    filtered_headers = {
        k: v for k, v in request.headers.items() if k.lower() in MCP_SERVER_SSE_HEADERS
    }
    sse_header_attributes = SseHeaderAttributes.from_request_headers(request.headers)

    async def event_generator():
        """
        Generate a merged stream of MCP server events and pending error messages.
        The pending error messages are added in the POST messages handler.
        This function runs in a loop, yielding events as they arrive.
        """
        mcp_server_events_queue = asyncio.Queue()
        pending_error_messages_queue = asyncio.Queue()
        tasks = set()
        session_id = None

        try:
            # MCP Server Events Processor
            async def process_mcp_server_events():
                """Connect to MCP server and process its events."""
                nonlocal session_id

                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(CLIENT_TIMEOUT)
                ) as client:
                    try:
                        async with aconnect_sse(
                            client,
                            "GET",
                            mcp_server_sse_endpoint,
                            headers=filtered_headers,
                            params=query_params,
                        ) as event_source:
                            if event_source.response.status_code != 200:
                                error_content = await event_source.response.aread()
                                raise HTTPException(
                                    status_code=event_source.response.status_code,
                                    detail=error_content,
                                )

                            async for sse in event_source.aiter_sse():
                                if sse.event == "endpoint":
                                    (
                                        event_bytes,
                                        extracted_id,
                                    ) = await _handle_endpoint_event(
                                        sse, sse_header_attributes
                                    )
                                    session_id = extracted_id

                                    if (
                                        session_id
                                        and "process_error_messages_task"
                                        not in locals()
                                    ):
                                        process_error_messages_task = (
                                            asyncio.create_task(
                                                _check_for_pending_error_messages(
                                                    session_id,
                                                    pending_error_messages_queue,
                                                )
                                            )
                                        )
                                        tasks.add(process_error_messages_task)
                                        process_error_messages_task.add_done_callback(
                                            tasks.discard
                                        )

                                elif sse.event == "message" and session_id:
                                    # Process message event
                                    event_bytes = await _handle_message_event(
                                        session_id, sse
                                    )
                                else:
                                    # Pass through other event types
                                    # pylint: disable=line-too-long
                                    event_bytes = f"event: {sse.event}\ndata: {sse.data}\n\n".encode(
                                        "utf-8"
                                    )

                                # Put the processed event in the queue
                                await mcp_server_events_queue.put(event_bytes)

                    except httpx.StreamClosed as e:
                        print(f"Server stream closed: {e}", flush=True)
                    except Exception as e:
                        print(f"Error processing server events: {e}", flush=True)

            # Start server events processor
            mcp_server_events_task = asyncio.create_task(process_mcp_server_events())
            tasks.add(mcp_server_events_task)
            mcp_server_events_task.add_done_callback(tasks.discard)

            # Main event loop: merge MCP server events and pending error messages
            while True:
                # Create futures for both queues
                mcp_server_event_future = asyncio.create_task(
                    mcp_server_events_queue.get()
                )
                pending_error_message_future = asyncio.create_task(
                    pending_error_messages_queue.get()
                )

                # Wait for either queue to have an item, with timeout
                done, pending = await asyncio.wait(
                    [mcp_server_event_future, pending_error_message_future],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.25,
                )

                for future in pending:
                    future.cancel()

                # Timeout occurred and no future completed.
                if not done:
                    continue

                for future in done:
                    try:
                        event = await future
                        yield event
                    except asyncio.CancelledError:
                        # Future was cancelled, continue
                        continue

        finally:
            # Clean up all tasks
            for task in tasks:
                task.cancel()

            # Wait for all tasks to complete
            if tasks:
                await asyncio.wait(tasks, timeout=2)

    # Return the streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Proxied-By": "mcp-gateway", **response_headers},
    )


async def _hook_tool_call(session_id: str, request_json: dict) -> Tuple[dict, bool]:
    """
    Hook to process the request JSON before sending it to the MCP server.

    Args:
        session_id (str): The session ID associated with the request.
        request_json (dict): The request JSON to be processed.
    """
    tool_call = {
        "id": f"call_{request_json.get('id')}",
        "type": "function",
        "function": {
            "name": request_json.get(MCP_PARAMS).get("name"),
            "arguments": request_json.get(MCP_PARAMS).get("arguments"),
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
        and _check_if_new_errors(session_id, guardrails_result)
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
            "id": request_json.get("id"),
            "error": {
                "code": -32600,
                "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                % guardrails_result["errors"],
            },
        }, True
    # Push trace to the explorer - don't block on its response
    await session_store.add_message_to_session(session_id, message, guardrails_result)
    return request_json, False


async def _hook_tool_call_response(session_id: str, response_json: dict, is_tools_list=False) -> dict:
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
        and _check_if_new_errors(session_id, guardrails_result)
    ):
        blocked = True
        # If the request is blocked, return a message indicating the block reason.
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
                            # no parameters
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
                }
            }

    # Push trace to the explorer - don't block on its response
    asyncio.create_task(
        session_store.add_message_to_session(session_id, message, guardrails_result)
    )
    return result, blocked


def _convert_localhost_to_docker_host(mcp_server_base_url: str) -> str:
    """
    Convert localhost or 127.0.0.1 in an address to host.docker.internal

    Args:
        mcp_server_base_url (str): The original server address from the header

    Returns:
        str: Modified server address with localhost references changed to host.docker.internal
    """
    # check if we are running in a docker container
    if not os.environ.get("DOCKER_ENV"):
        return mcp_server_base_url

    if "localhost" in mcp_server_base_url or "127.0.0.1" in mcp_server_base_url:
        # Replace localhost or 127.0.0.1 with host.docker.internal
        modified_address = re.sub(
            r"(https?://)(?:localhost|127\.0\.0\.1)(\b|:)",
            r"\1host.docker.internal\2",
            mcp_server_base_url,
        )
        return modified_address

    return mcp_server_base_url


async def _handle_endpoint_event(
    sse: ServerSentEvent, sse_header_attributes: SseHeaderAttributes
) -> Tuple[bytes, str]:
    """
    Handle the endpoint event type and modify the data accordingly.
    For endpoint events, we need to rewrite the endpoint to use our gateway.

    Args:
        sse (ServerSentEvent): The original SSE object.
        sse_header_attributes (SseHeaderAttributes): The header attributes from the request.

    Returns:
        bytes: Modified SSE data as bytes.
        str: session_id extracted from the data.
    """
    # Extract session_id
    match = re.search(r"session_id=([^&\s]+)", sse.data)
    if match:
        session_id = match.group(1)
        # Initialize this session in our store if needed
        if not session_store.session_exists(session_id):
            await session_store.initialize_session(session_id, sse_header_attributes)

    # Rewrite the endpoint to use our gateway
    modified_data = sse.data.replace(
        "/messages/?session_id=",
        "/api/v1/gateway/mcp/sse/messages/?session_id=",
    )
    event_bytes = f"event: {sse.event}\ndata: {modified_data}\n\n".encode("utf-8")
    return event_bytes, session_id


async def _handle_message_event(session_id: str, sse: ServerSentEvent) -> bytes:
    """
    Handle the message event type.

    Args:
        session_id (str): The session ID associated with the request.
        sse (ServerSentEvent): The original SSE object.
    """
    event_bytes = f"event: {sse.event}\ndata: {sse.data}\n\n".encode("utf-8")
    session = session_store.get_session(session_id)
    try:
        response_json = json.loads(sse.data)

        if response_json.get(MCP_RESULT) and response_json.get(MCP_RESULT).get(
            MCP_SERVER_INFO
        ):
            session.metadata["mcp_server_name"] = (
                response_json.get(MCP_RESULT).get(MCP_SERVER_INFO).get("name", "")
            )

        method = session.id_to_method_mapping.get(response_json.get("id"))
        if method == MCP_TOOL_CALL:
            hook_tool_call_response, blocked = await _hook_tool_call_response(
                session_id=session_id,
                response_json=response_json,
            )
            # Update the event bytes with hook_tool_call_response.
            # hook_tool_call_response is same as response_json if no guardrail is violated.
            # If guardrail is violated, it contains the error message.
            # pylint: disable=line-too-long
            if blocked:
                event_bytes = f"event: {sse.event}\ndata: {json.dumps(hook_tool_call_response)}\n\n".encode(
                    "utf-8"
                )
        elif method == MCP_LIST_TOOLS:
            # store tools in metadata
            session_store.get_session(session_id).metadata["tools"] = response_json.get(
                MCP_RESULT
            ).get("tools")
            # store tools/list tool call in trace
            hook_tool_call_response, blocked = await _hook_tool_call_response(
                session_id=session_id,
                response_json={
                    "id": response_json.get("id"),
                    "result": {
                        "content": json.dumps(response_json.get(MCP_RESULT).get("tools")),
                        "tools": response_json.get(MCP_RESULT).get("tools"),
                    }
                },
                is_tools_list=True,
            )
            # Update the event bytes with hook_tool_call_response.
            # hook_tool_call_response is same as response_json if no guardrail is violated.
            # If guardrail is violated, it contains the error message.
            # pylint: disable=line-too-long
            if blocked:
                event_bytes = f"event: {sse.event}\ndata: {json.dumps(hook_tool_call_response)}\n\n".encode(
                    "utf-8"
                )

    except json.JSONDecodeError as e:
        print(
            f"[MCP SSE] Error parsing message JSON: {e}",
            flush=True,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(
            f"[MCP SSE] Error processing message: {e}",
            flush=True,
        )
    return event_bytes


def _check_if_new_errors(session_id: str, guardrails_result: dict) -> bool:
    """Checks if there are new errors in the guardrails result."""
    session = session_store.get_session(session_id)
    annotations = create_annotations_from_guardrails_errors(
        guardrails_result.get("errors", [])
    )
    for annotation in annotations:
        if annotation not in session.annotations:
            return True
    return False


async def _check_for_pending_error_messages(
    session_id: str, pending_error_messages_queue: asyncio.Queue
):
    """Periodically check for and enqueue pending error messages."""
    try:
        while True:
            try:
                session = session_store.get_session(session_id)
                error_messages = await session.get_pending_error_messages()

                for error_message in error_messages:
                    error_bytes = (
                        f"event: message\ndata: {json.dumps(error_message)}\n\n".encode(
                            "utf-8"
                        )
                    )
                    await pending_error_messages_queue.put(error_bytes)

                await asyncio.sleep(1)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error checking for messages: {e}", flush=True)
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        return

"""Gateway service to forward requests to the MCP SSE servers"""

import asyncio
import json
import re
from typing import Tuple

import httpx
from httpx_sse import aconnect_sse, ServerSentEvent
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from gateway.common.constants import (
    CLIENT_TIMEOUT,
    MCP_METHOD,
    MCP_TOOL_CALL,
    MCP_LIST_TOOLS,
    UTF_8,
)
from gateway.mcp.mcp_sessions_manager import (
    McpSessionsManager,
    McpAttributes,
)
from gateway.mcp.utils import (
    get_mcp_server_base_url,
    hook_tool_call,
    intercept_response,
    update_mcp_server_in_session_metadata,
    update_session_from_request,
)

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
MCP_SERVER_BASE_URL_HEADER = "mcp-server-base-url"

gateway = APIRouter()
session_store = McpSessionsManager()


@gateway.post("/mcp/sse/messages/")
async def mcp_post_sse_gateway(
    request: Request,
) -> Response:
    """Proxy calls to the MCP Server tools"""
    query_params = dict(request.query_params)
    if not query_params.get("session_id"):
        raise HTTPException(
            status_code=400,
            detail="Missing 'session_id' query parameter",
        )
    if not session_store.session_exists(query_params.get("session_id")):
        raise HTTPException(
            status_code=400,
            detail="Session does not exist",
        )

    session_id = query_params.get("session_id")
    mcp_server_messages_endpoint = (
        get_mcp_server_base_url(request) + "/messages/?" + session_id
    )
    request_body_bytes = await request.body()
    request_body = json.loads(request_body_bytes)
    session = session_store.get_session(session_id)
    update_session_from_request(session, request_body)

    if request_body.get(MCP_METHOD) == MCP_TOOL_CALL:
        # Intercept and potentially block the request
        hook_tool_call_result, is_blocked = await hook_tool_call(
            session_id=session_id,
            session_store=session_store,
            request_body=request_body,
        )
        if is_blocked:
            # Add the error message to the session.
            # The error message is sent back to the client using the SSE stream.
            await session.add_pending_error_message(hook_tool_call_result)
            return Response(content="Accepted", status_code=202)
    elif request_body.get(MCP_METHOD) == MCP_LIST_TOOLS:
        # Intercept and potentially block the request
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
                json=request_body,
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
    mcp_server_sse_endpoint = get_mcp_server_base_url(request) + "/sse"

    query_params = dict(request.query_params)
    response_headers = {}
    filtered_headers = {
        k: v for k, v in request.headers.items() if k.lower() in MCP_SERVER_SSE_HEADERS
    }
    sse_header_attributes = McpAttributes.from_request_headers(request.headers)

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
                            response_headers.update(
                                dict(event_source.response.headers.items())
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
                                        UTF_8
                                    )

                                # Put the processed event in the queue
                                await mcp_server_events_queue.put(event_bytes)

                    except httpx.StreamClosed as e:
                        print(f"Server stream closed: {e}", flush=True)
                    except Exception as e:  # pylint: disable=broad-except
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


async def _handle_endpoint_event(
    sse: ServerSentEvent, sse_header_attributes: McpAttributes
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
    event_bytes = f"event: {sse.event}\ndata: {modified_data}\n\n".encode(UTF_8)
    return event_bytes, session_id


async def _handle_message_event(session_id: str, sse: ServerSentEvent) -> bytes:
    """
    Handle the message event type.

    Args:
        session_id (str): The session ID associated with the request.
        sse (ServerSentEvent): The original SSE object.
    """
    event_bytes = f"event: {sse.event}\ndata: {sse.data}\n\n".encode(UTF_8)
    session = session_store.get_session(session_id)
    try:
        response_body = json.loads(sse.data)
        update_mcp_server_in_session_metadata(session, response_body)

        intercept_response_result, is_blocked = await intercept_response(
            session_id=session_id,
            session_store=session_store,
            response_body=response_body,
        )
        if is_blocked:
            event_bytes = f"event: {sse.event}\ndata: {json.dumps(intercept_response_result)}\n\n".encode(
                UTF_8
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
                            UTF_8
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

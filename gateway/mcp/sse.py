"""Gateway service to forward requests to the MCP SSE servers"""

import asyncio
import json
import re
from typing import Any, AsyncGenerator

import httpx
from httpx_sse import aconnect_sse, ServerSentEvent
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from gateway.common.constants import CLIENT_TIMEOUT, CONTENT_TYPE_EVENT_STREAM
from gateway.mcp.constants import MCP_CUSTOM_HEADER_PREFIX, UTF_8
from gateway.mcp.mcp_sessions_manager import (
    McpSessionsManager,
    McpAttributes,
)
from gateway.mcp.mcp_transport_base import McpTransportBase

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
mcp_sessions_manager = McpSessionsManager()


@gateway.post("/mcp/sse/messages/")
async def mcp_post_sse_gateway(request: Request) -> Response:
    """Proxy calls to the MCP Server using SSE transport strategy."""
    return await create_sse_transport_and_handle_post(request, mcp_sessions_manager)


@gateway.get("/mcp/sse")
async def mcp_get_sse_gateway(request: Request) -> StreamingResponse:
    """Proxy calls to the MCP Server using SSE transport strategy."""
    return await create_sse_transport_and_handle_stream(request, mcp_sessions_manager)


async def create_sse_transport_and_handle_post(
    request: Request, session_store: McpSessionsManager
) -> Response:
    """Integration function for SSE POST route."""
    query_params = dict(request.query_params)
    session_id = query_params.get("session_id")

    if not session_id:
        raise HTTPException(
            status_code=400, detail="Missing 'session_id' query parameter"
        )
    if not session_store.session_exists(session_id):
        raise HTTPException(status_code=400, detail="Session does not exist")

    request_body = json.loads(await request.body())
    return await SseTransport(session_store).handle_post_request(
        request, session_id, request_body
    )


async def create_sse_transport_and_handle_stream(
    request: Request, session_store: McpSessionsManager
) -> StreamingResponse:
    """Integration function for SSE GET route."""
    return await SseTransport(session_store).handle_sse_stream(request)


class SseTransport(McpTransportBase):
    """
    Server-Sent Events transport implementation for MCP communication.
    Handles HTTP-based SSE communication with message queuing.
    """

    async def initialize_session(
        self,
        **kwargs,
    ) -> str:
        """Initialize or get existing SSE session."""
        session_id: str | None = kwargs.get("session_id", None)
        session_attributes: McpAttributes | None = kwargs.get(
            "session_attributes", None
        )
        if session_id and self.session_store.session_exists(session_id):
            return session_id

        if not session_id:
            raise ValueError("Session ID is required for SSE transport")

        if not self.session_store.session_exists(session_id):
            if not session_attributes:
                raise ValueError("Session attributes required for new session")
            await self.session_store.initialize_session(session_id, session_attributes)

        return session_id

    async def handle_post_request(
        self, request: Request, session_id: str, request_body: dict[str, Any]
    ) -> Response:
        """Handle POST request to SSE endpoint."""
        session = self.session_store.get_session(session_id)

        processed_request, is_blocked = await self.process_outgoing_request(
            session_id, request_body
        )

        if is_blocked:
            # Add the error message to the session for SSE delivery
            await session.add_pending_error_message(processed_request)
            return Response(content="Accepted", status_code=202)

        # Forward to MCP server
        mcp_server_base_url = self.get_mcp_server_base_url(request)
        mcp_server_messages_endpoint = f"{mcp_server_base_url}/messages/?{session_id}"

        # Filter headers for MCP server
        filtered_headers = {}
        for k, v in request.headers.items():
            if k.startswith(MCP_CUSTOM_HEADER_PREFIX):
                filtered_headers[k.removeprefix(MCP_CUSTOM_HEADER_PREFIX)] = v
            if k.lower() in MCP_SERVER_POST_HEADERS:
                filtered_headers[k] = v

        async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
            try:
                response = await client.post(
                    url=mcp_server_messages_endpoint,
                    headers=filtered_headers,
                    json=request_body,
                    params=dict(request.query_params),
                )
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={"X-Proxied-By": "mcp-gateway", **response.headers},
                )
            except httpx.RequestError as e:
                print(f"[MCP POST] Request error: {str(e)}")
                raise HTTPException(status_code=500, detail="Request error") from e

    async def handle_sse_stream(self, request: Request) -> StreamingResponse:
        """Handle SSE streaming connection."""
        mcp_server_base_url = self.get_mcp_server_base_url(request)
        mcp_server_sse_endpoint = f"{mcp_server_base_url}/sse"

        query_params = dict(request.query_params)
        response_headers = {}

        # Filter headers for SSE
        filtered_headers = {}
        for k, v in request.headers.items():
            if k.startswith(MCP_CUSTOM_HEADER_PREFIX):
                filtered_headers[k.removeprefix(MCP_CUSTOM_HEADER_PREFIX)] = v
            if k.lower() in MCP_SERVER_SSE_HEADERS:
                filtered_headers[k] = v

        sse_header_attributes = McpAttributes.from_request_headers(request.headers)

        async def event_generator() -> AsyncGenerator[bytes, None]:
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
                                        ) = await self._handle_endpoint_event(
                                            sse, sse_header_attributes
                                        )
                                        session_id = extracted_id

                                        if (
                                            session_id
                                            and "process_error_messages_task"
                                            not in locals()
                                        ):
                                            process_error_messages_task = asyncio.create_task(
                                                self._check_for_pending_error_messages(
                                                    session_id,
                                                    pending_error_messages_queue,
                                                )
                                            )
                                            tasks.add(process_error_messages_task)
                                            process_error_messages_task.add_done_callback(
                                                tasks.discard
                                            )

                                    elif sse.event == "message" and session_id:
                                        event_bytes = await self._handle_message_event(
                                            session_id, sse
                                        )
                                    else:
                                        event_bytes = f"event: {sse.event}\ndata: {sse.data}\n\n".encode(
                                            UTF_8
                                        )

                                    await mcp_server_events_queue.put(event_bytes)

                        except httpx.StreamClosed as e:
                            print(f"Server stream closed: {e}")
                        except Exception as e:  # pylint: disable=broad-except
                            print(f"Error processing server events: {e}")

                # Start server events processor
                mcp_server_events_task = asyncio.create_task(
                    process_mcp_server_events()
                )
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

        return StreamingResponse(
            event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
            headers={"X-Proxied-By": "mcp-gateway", **response_headers},
        )

    async def handle_communication(self, **kwargs) -> StreamingResponse:
        """Main communication handler for SSE transport."""
        return await self.handle_sse_stream(kwargs.get("request"))

    async def _handle_endpoint_event(
        self, sse: ServerSentEvent, sse_header_attributes: McpAttributes
    ) -> tuple[bytes, str]:
        """Handle endpoint event and initialize session if needed."""
        match = re.search(r"session_id=([^&\s]+)", sse.data)
        session_id = match.group(1) if match else None

        if session_id:
            # Initialize session if needed
            await self.initialize_session(
                session_id=session_id, session_attributes=sse_header_attributes
            )

        # Rewrite endpoint to use our gateway
        modified_data = sse.data.replace(
            "/messages/?session_id=",
            "/api/v1/gateway/mcp/sse/messages/?session_id=",
        )
        event_bytes = f"event: {sse.event}\ndata: {modified_data}\n\n".encode(UTF_8)
        return event_bytes, session_id

    async def _handle_message_event(
        self, session_id: str, sse: ServerSentEvent
    ) -> bytes:
        """Handle message event with guardrails processing."""
        try:
            response_body = json.loads(sse.data)

            # Process response through guardrails
            processed_response, is_blocked = await self.process_incoming_response(
                session_id, response_body
            )

            event_bytes = f"event: {sse.event}\ndata: {sse.data}\n\n".encode(UTF_8)
            if is_blocked:
                event_bytes = f"event: {sse.event}\ndata: {json.dumps(processed_response)}\n\n".encode(
                    UTF_8
                )

            return event_bytes
        except json.JSONDecodeError as e:
            print(f"[MCP SSE] Error parsing message JSON: {e}")
            return f"event: {sse.event}\ndata: {sse.data}\n\n".encode(UTF_8)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[MCP SSE] Error processing message: {e}")
            return f"event: {sse.event}\ndata: {sse.data}\n\n".encode(UTF_8)

    async def _check_for_pending_error_messages(
        self, session_id: str, pending_error_messages_queue: asyncio.Queue
    ):
        """Periodically check for and enqueue pending error messages."""
        try:
            while True:
                try:
                    session = self.session_store.get_session(session_id)
                    error_messages = await session.get_pending_error_messages()

                    for error_message in error_messages:
                        error_bytes = f"event: message\ndata: {json.dumps(error_message)}\n\n".encode(
                            UTF_8
                        )
                        await pending_error_messages_queue.put(error_bytes)

                    await asyncio.sleep(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error checking for messages: {e}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            return

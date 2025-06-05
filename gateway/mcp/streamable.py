"""Gateway service to forward requests to the MCP Streamable HTTP servers"""

import json
from typing import Any

import httpx
from httpx_sse import aconnect_sse
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from gateway.common.constants import (
    CLIENT_TIMEOUT,
    CONTENT_TYPE_HEADER,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_EVENT_STREAM,
)
from gateway.mcp.constants import (
    INVARIANT_SESSION_ID_PREFIX,
    MCP_CUSTOM_HEADER_PREFIX,
    UTF_8,
)
from gateway.mcp.mcp_sessions_manager import (
    McpSessionsManager,
    McpAttributes,
)
from gateway.mcp.mcp_transport_base import McpTransportBase

gateway = APIRouter()
mcp_sessions_manager = McpSessionsManager()

MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_SERVER_POST_AND_DELETE_HEADERS = {
    "connection",
    "accept",
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
async def mcp_post_streamable_gateway(
    request: Request,
):
    """Forward a POST request to the MCP Streamable server using transport strategy."""
    return await create_streamable_transport_and_handle_request(
        request, "POST", mcp_sessions_manager
    )


@gateway.get("/mcp/streamable")
async def mcp_get_streamable_gateway(request: Request) -> StreamingResponse:
    """Forward a GET request to the MCP Streamable server using transport strategy."""
    return await create_streamable_transport_and_handle_request(
        request, "GET", mcp_sessions_manager
    )


@gateway.delete("/mcp/streamable")
async def mcp_delete_streamable_gateway(request: Request) -> Response:
    """Forward a DELETE request to the MCP Streamable server using transport strategy."""
    return await create_streamable_transport_and_handle_request(
        request, "DELETE", mcp_sessions_manager
    )


async def create_streamable_transport_and_handle_request(
    request: Request, method: str, session_store: McpSessionsManager
) -> Response | StreamingResponse:
    """Integration function for streamable routes."""
    streamable_transport = StreamableTransport(session_store)
    return await streamable_transport.handle_communication(
        request=request, method=method
    )


class StreamableTransport(McpTransportBase):
    """
    Streamable HTTP transport implementation for MCP communication.
    Handles HTTP POST/GET/DELETE requests with JSON and streaming responses.
    """

    async def initialize_session(
        self,
        **kwargs,
    ) -> str:
        """Initialize streamable HTTP session."""
        session_id: str | None = kwargs.get("session_id", None)
        session_attributes: McpAttributes | None = kwargs.get(
            "session_attributes", None
        )
        is_initialization_request: bool = kwargs.get("is_initialization_request", False)
        if session_id and self.session_store.session_exists(session_id):
            return session_id

        if is_initialization_request and not session_id:
            session_id = self.generate_session_id()

        if (
            session_id
            and not self.session_store.session_exists(session_id)
            and session_attributes
        ):
            await self.session_store.initialize_session(session_id, session_attributes)

        return session_id

    async def handle_post_request(
        self, request: Request, request_body: dict[str, Any]
    ) -> Response | StreamingResponse:
        """Handle POST request to streamable endpoint."""
        session_attributes = McpAttributes.from_request_headers(request.headers)
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)
        is_initialization_request = self._is_initialization_request(request_body)

        # Handle session initialization
        if session_id:
            self.update_tool_call_id_in_session(
                self.session_store.get_session(session_id), request_body
            )
        elif is_initialization_request:
            session_id = await self.initialize_session(
                session_attributes=session_attributes, is_initialization_request=True
            )

        # Process request if not initialization
        if not is_initialization_request:
            request_interception_result = await self._process_non_init_request(
                session_id, request_body
            )
            if request_interception_result:
                return request_interception_result

        # Forward to MCP server
        return await self._forward_to_mcp_server(
            request,
            request_body,
            session_id,
            session_attributes,
            is_initialization_request,
        )

    async def handle_get_request(self, request: Request) -> StreamingResponse:
        """Handle GET request for server-initiated communication."""
        mcp_server_endpoint = self._get_mcp_server_endpoint(request)
        response_headers = {}

        filtered_headers = {}
        for k, v in request.headers.items():
            if k.startswith(MCP_CUSTOM_HEADER_PREFIX):
                filtered_headers[k.removeprefix(MCP_CUSTOM_HEADER_PREFIX)] = v
            if k.lower() in MCP_SERVER_GET_HEADERS:
                filtered_headers[k] = v

        async def event_generator():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(CLIENT_TIMEOUT)
            ) as client:
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

                        response_headers.update(
                            dict(event_source.response.headers.items())
                        )

                        async for sse in event_source.aiter_sse():
                            yield sse

                except httpx.StreamClosed as e:
                    print(f"Server stream closed: {e}")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error processing server events: {e}")

        return StreamingResponse(
            event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
            headers={"X-Proxied-By": "mcp-gateway", **response_headers},
        )

    async def handle_delete_request(self, request: Request) -> Response:
        """Handle DELETE request for session termination."""
        session_id = self._get_session_id(request)

        if not self.session_store.session_exists(session_id):
            raise HTTPException(status_code=400, detail="Session does not exist")

        if session_id.startswith(INVARIANT_SESSION_ID_PREFIX):
            return Response(
                content="", status_code=200, headers={"X-Proxied-By": "mcp-gateway"}
            )

        mcp_server_endpoint = self._get_mcp_server_endpoint(request)

        async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
            try:
                response = await client.delete(
                    url=mcp_server_endpoint,
                    headers=self._get_headers_for_mcp_post_and_delete(request),
                )
                await self.session_store.cleanup_session_lock(session_id)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={"X-Proxied-By": "mcp-gateway", **response.headers},
                )
            except httpx.RequestError as e:
                print(f"[MCP DELETE] Request error: {str(e)}")
                raise HTTPException(status_code=500, detail="Request error") from e

    async def handle_communication(self, **kwargs) -> Response | StreamingResponse:
        """Main communication handler for streamable transport."""
        request = kwargs.get("request")
        method = kwargs.get("method", "POST")

        if method == "POST":
            request_body = json.loads(await request.body())
            return await self.handle_post_request(request, request_body)
        elif method == "GET":
            return await self.handle_get_request(request)
        elif method == "DELETE":
            return await self.handle_delete_request(request)
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")

    async def _process_non_init_request(
        self, session_id: str, request_body: dict[str, Any]
    ) -> Response | None:
        """Process non-initialization requests for guardrails."""
        processed_request, is_blocked = await self.process_outgoing_request(
            session_id, request_body
        )

        if is_blocked:
            return Response(
                content=json.dumps(processed_request),
                status_code=400,
                media_type=CONTENT_TYPE_JSON,
            )
        return None

    async def _forward_to_mcp_server(
        self,
        request: Request,
        request_body: dict[str, Any],
        session_id: str,
        session_attributes: McpAttributes,
        is_initialization_request: bool,
    ) -> Response | StreamingResponse:
        """Forward request to MCP server and handle response."""
        async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
            try:
                response = await client.post(
                    url=self._get_mcp_server_endpoint(request),
                    headers=self._get_headers_for_mcp_post_and_delete(request),
                    content=json.dumps(request_body).encode(),
                    follow_redirects=True,
                )

                # Handle session ID from MCP server response
                resp_session_id = response.headers.get(MCP_SESSION_ID_HEADER)

                if resp_session_id:
                    if not self.session_store.session_exists(resp_session_id):
                        await self.session_store.initialize_session(
                            resp_session_id, session_attributes
                        )
                    session_id = resp_session_id
                elif (
                    is_initialization_request
                    and not self.session_store.session_exists(session_id)
                ):
                    await self.session_store.initialize_session(
                        session_id, session_attributes
                    )

                # Update client info for initialization requests
                if is_initialization_request:
                    self.update_mcp_client_info_in_session(
                        self.session_store.get_session(session_id), request_body
                    )

                # Handle response based on content type
                if response.headers.get(CONTENT_TYPE_HEADER) == CONTENT_TYPE_JSON:
                    return await self._handle_json_response(
                        session_id, is_initialization_request, response
                    )
                else:
                    return await self._handle_streaming_response(
                        session_id, is_initialization_request, response
                    )

            except httpx.RequestError as e:
                print(f"[MCP POST] Request error: {str(e)}")
                raise HTTPException(status_code=500, detail="Request error") from e

    async def _handle_json_response(
        self, session_id: str, is_initialization_request: bool, response: httpx.Response
    ) -> Response:
        """Handle JSON response from MCP server."""
        response_content = response.content
        response_body = (
            json.loads(response_content.decode(UTF_8)) if response_content else {}
        )

        if response_body:
            self._update_mcp_response_info_in_session(session_id, response_body, True)

        response_code = response.status_code

        if not is_initialization_request and response_body:
            processed_response, blocked = await self.process_incoming_response(
                session_id, response_body
            )
            if blocked:
                response_content = json.dumps(processed_response).encode(UTF_8)
                response_code = 400

        # Build response headers
        response_headers = {"X-Proxied-By": "mcp-gateway", **response.headers}
        if MCP_SESSION_ID_HEADER not in response.headers:
            response_headers[MCP_SESSION_ID_HEADER] = session_id

        return Response(
            content=response_content,
            status_code=response_code,
            headers=response_headers,
        )

    async def _handle_streaming_response(
        self, session_id: str, is_initialization_request: bool, response: httpx.Response
    ) -> StreamingResponse:
        """Handle streaming response from MCP server."""

        async def event_generator():
            buffer = ""
            async for line in response.aiter_lines():
                stripped_line = line.strip()
                if not stripped_line:
                    break

                if buffer:
                    response_body = json.loads(stripped_line.split("data: ")[1].strip())

                    if not is_initialization_request:
                        (
                            processed_response,
                            blocked,
                        ) = await self.process_incoming_response(
                            session_id, response_body
                        )
                        if blocked:
                            yield f"{buffer}\ndata: {json.dumps(processed_response)}\n\n"
                            break
                    else:
                        self._update_mcp_response_info_in_session(
                            session_id, response_body, False
                        )

                    yield f"{buffer}\n{stripped_line}\n\n"
                    buffer = ""
                else:
                    buffer = stripped_line

        # Build response headers
        response_headers = {"X-Proxied-By": "mcp-gateway", **response.headers}
        if MCP_SESSION_ID_HEADER not in response.headers:
            response_headers[MCP_SESSION_ID_HEADER] = session_id

        return StreamingResponse(
            event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
            headers=response_headers,
        )

    def _update_mcp_response_info_in_session(
        self, session_id: str, response_body: dict, is_json_response: bool
    ) -> None:
        """Update MCP response info in session metadata."""
        session = self.session_store.get_session(session_id)
        self.update_mcp_server_in_session_metadata(session, response_body)
        session.attributes.metadata["is_stateless_http_server"] = session_id.startswith(
            INVARIANT_SESSION_ID_PREFIX
        )
        session.attributes.metadata["server_response_type"] = (
            "json" if is_json_response else "sse"
        )

    def _get_headers_for_mcp_post_and_delete(self, request: Request) -> dict:
        """Get filtered headers for MCP server requests."""
        filtered_headers = {}
        for k, v in request.headers.items():
            if k.startswith(MCP_CUSTOM_HEADER_PREFIX):
                filtered_headers[k.removeprefix(MCP_CUSTOM_HEADER_PREFIX)] = v
            if k.lower() in MCP_SERVER_POST_AND_DELETE_HEADERS and not (
                k.lower() == MCP_SESSION_ID_HEADER
                and v.startswith(INVARIANT_SESSION_ID_PREFIX)
            ):
                filtered_headers[k] = v
        return filtered_headers

    def _get_session_id(self, request: Request) -> str:
        """Extract session ID from request headers."""
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing mcp-session-id header")
        return session_id

    def _get_mcp_server_endpoint(self, request: Request) -> str:
        """Get MCP server endpoint URL."""
        return self.get_mcp_server_base_url(request) + "/mcp/"

    def _is_initialization_request(self, request_data: dict[str, Any]) -> bool:
        """Check if request is an initialization request."""
        return (
            request_data.get("method") in ["initialize", "notifications/initialized"]
            and "jsonrpc" in request_data
        )

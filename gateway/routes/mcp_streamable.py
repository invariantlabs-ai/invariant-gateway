"""Gateway service to forward requests to the MCP Streamable HTTP servers"""

import json

from gateway.common.constants import CLIENT_TIMEOUT
from gateway.common.mcp_sessions_manager import McpSessionsManager, SseHeaderAttributes
from gateway.common.mcp_utils import get_mcp_server_base_url

import httpx

from httpx_sse import aconnect_sse
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse


gateway = APIRouter()
session_store = McpSessionsManager()

MCP_SESSION_ID_HEADER = "mcp-session-id"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"
MCP_SERVER_POST_HEADERS = {
    "connection",
    "accept",
    "content-length",
    "content-type",
    MCP_SESSION_ID_HEADER,
}
MCP_SERVER_SSE_HEADERS = {
    "connection",
    "accept",
    "cache-control",
    MCP_SESSION_ID_HEADER,
}


def get_session_id(request: Request) -> str:
    """Extract the session ID from request headers."""
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {MCP_SESSION_ID_HEADER} header",
        )
    return session_id


def get_mcp_server_endpoint(request: Request) -> str:
    """
    Extract the MCP server endpoint from the request headers.
    """
    return get_mcp_server_base_url(request) + "/mcp/"


@gateway.post("/mcp/streamable")
async def mcp_post_streamable_gateway(
    request: Request,
) -> StreamingResponse:
    """
    Forward a POST request to the MCP Streamable server.
    """
    body = await request.body()
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)

    # Determine if this is an initialization request, only for our session tracking
    try:
        raw_message = json.loads(body)
        is_initialization_request = (
            isinstance(raw_message, dict)
            and raw_message.get("method") == "initialize"
            and "jsonrpc" in raw_message
        )
    except json.JSONDecodeError:
        # Let the server handle the validation error
        pass

    mcp_server_endpoint = get_mcp_server_endpoint(request)
    filtered_headers = {
        k: v for k, v in request.headers.items() if k.lower() in MCP_SERVER_POST_HEADERS
    }
    if (
        session_id
        and not is_initialization_request
        and not session_store.session_exists(session_id)
    ):
        raise HTTPException(status_code=404, detail="Invalid or expired session ID")
    sse_header_attributes = SseHeaderAttributes.from_request_headers(request.headers)

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        try:
            response = await client.post(
                url=mcp_server_endpoint,
                headers=filtered_headers,
                content=body,
                follow_redirects=True,
            )

            # If we received a session ID from server, register it in our session store
            # This happens in initialization responses
            resp_session_id = response.headers.get(MCP_SESSION_ID_HEADER)
            if resp_session_id and not session_store.session_exists(resp_session_id):
                await session_store.initialize_session(
                    resp_session_id, sse_header_attributes
                )

            # If the response is JSON, return it directly
            if response.headers.get("content-type", "") == CONTENT_TYPE_JSON:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={"X-Proxied-By": "mcp-gateway", **response.headers},
                )

            # Else return SSE streaming response
            async def event_generator():
                # Events have two parts:
                # 1. event: {type} -> contains the type of event
                # 2. data: {data} -> contains the actual message
                # We are reading line by line so we need to buffer so that we can
                # send the entire event (with both type and data) together.
                # Once we receive an empty line, we end the stream.
                buffer = ""
                async for line in response.aiter_lines():
                    if line.strip():
                        if buffer:
                            complete_event = buffer + "\n" + line + "\n\n"
                            yield complete_event
                            # Clear the buffer for the next event
                            buffer = ""
                        else:
                            buffer = line
                    else:
                        # End stream here when line is empty.
                        break

            return StreamingResponse(
                event_generator(),
                media_type=CONTENT_TYPE_SSE,
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
    mcp_server_endpoint = get_mcp_server_endpoint(request)
    response_headers = {}
    filtered_headers = {
        k: v for k, v in request.headers.items() if k.lower() in MCP_SERVER_SSE_HEADERS
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
    session_id = get_session_id(request)
    if not session_store.session_exists(session_id):
        raise HTTPException(
            status_code=400,
            detail="Session does not exist",
        )
    mcp_server_endpoint = get_mcp_server_endpoint(request)

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        try:
            response = await client.delete(
                url=mcp_server_endpoint,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower()
                    in {
                        "connection",
                        "accept",
                        "content-length",
                        "content-type",
                        MCP_SESSION_ID_HEADER,
                    }
                },
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
            print(f"[MCP POST] Request error: {str(e)}")
            raise HTTPException(status_code=500, detail="Request error") from e
        except Exception as e:
            print(f"[MCP POST] Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Unexpected error") from e

"""Proxy service to forward requests to the Gemini APIs"""

import json
from typing import Any, Optional

import httpx
from common.config_manager import ProxyConfig, ProxyConfigManager
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from utils.constants import CLIENT_TIMEOUT, IGNORED_HEADERS

proxy = APIRouter()


@proxy.post("/gemini/{api_version}/models/{model}:{endpoint}")
@proxy.post("/{dataset_name}/gemini/{api_version}/models/{model}:{endpoint}")
async def gemini_generate_content_proxy(
    request: Request,
    api_version: str,
    model: str,
    endpoint: str,
    dataset_name: str = None,
    alt: str = Query(
        None, title="Response Format", description="Set to 'sse' for streaming"
    ),
    config: ProxyConfig = Depends(ProxyConfigManager.get_config),  # pylint: disable=unused-argument
) -> Response:
    """Proxy calls to the Gemini GenerateContent API"""
    if "generateContent" != endpoint and "streamGenerateContent" != endpoint:
        return Response(
            content="Invalid endpoint - the only endpoints supported are: \
            /api/v1/proxy/gemini/<version>/models/<model-name>:generateContent or \
            /api/v1/proxy/<dataset-name>/gemini/<version>models/<model-name>:generateContent",
            status_code=400,
        )
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    request_body_bytes = await request.body()
    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    gemini_api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:{endpoint}"
    if alt == "sse":
        gemini_api_url += "?alt=sse"
    gemini_request = client.build_request(
        "POST",
        gemini_api_url,
        content=request_body_bytes,
        headers=headers,
    )

    if alt == "sse" or endpoint == "streamGenerateContent":
        return await stream_response(
            client,
            gemini_request,
            dataset_name,
        )
    response = await client.send(gemini_request)
    return await handle_non_streaming_response(response, dataset_name)


async def stream_response(
    client: httpx.AsyncClient,
    gemini_request: httpx.Request,
    dataset_name: Optional[str],
) -> Response:
    """Handles streaming the Gemini response to the client"""

    response = await client.send(gemini_request, stream=True)
    if response.status_code != 200:
        error_content = await response.aread()
        try:
            error_json = json.loads(error_content.decode("utf-8"))
            error_detail = error_json.get("error", "Unknown error from Gemini API")
        except json.JSONDecodeError:
            error_detail = {"error": "Failed to parse Gemini error response"}
        raise HTTPException(status_code=response.status_code, detail=error_detail)

    async def event_generator() -> Any:
        async for chunk in response.aiter_bytes():
            chunk_text = chunk.decode().strip()
            if not chunk_text:
                continue

            # Yield chunk immediately to the client (proxy behavior)
            yield chunk

        # Send full merged response to the explorer
        if dataset_name:
            # Push to Explorer
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def handle_non_streaming_response(
    response: httpx.Response,
    dataset_name: Optional[str],
) -> Response:
    """Handles non-streaming Gemini responses"""
    try:
        json_response = response.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail="Invalid JSON response received from Gemini API",
        ) from e
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=json_response.get("error", "Unknown error from Gemini API"),
        )
    if dataset_name:
        # Push to Explorer
        pass

    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

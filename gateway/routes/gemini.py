"""Gateway service to forward requests to the Gemini APIs"""

import json
from typing import Any, Optional

import httpx
from common.config_manager import GatewayConfig, GatewayConfigManager
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from common.authorization import extract_authorization_from_headers
from converters.gemini_to_invariant import convert_request, convert_response
from integrations.explorer import push_trace

gateway = APIRouter()

GEMINI_AUTHORIZATION_HEADER = "x-goog-api-key"


@gateway.post("/gemini/{api_version}/models/{model}:{endpoint}")
async def gemini_generate_content_gateway(
    request: Request,
    api_version: str,
    model: str,
    endpoint: str,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
    alt: str = Query(
        None, title="Response Format", description="Set to 'sse' for streaming"
    ),
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),  # pylint: disable=unused-argument
) -> Response:
    """Proxy calls to the Gemini GenerateContent API"""
    if endpoint not in ["generateContent", "streamGenerateContent"]:
        return Response(
            content="Invalid endpoint - the only endpoints supported are: \
            /api/v1/gateway/gemini/<version>/models/<model-name>:generateContent or \
            /api/v1/gateway/<dataset-name>/gemini/<version>models/<model-name>:generateContent",
            status_code=400,
        )
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, gemini_api_key = extract_authorization_from_headers(
        request, dataset_name, GEMINI_AUTHORIZATION_HEADER
    )
    headers[GEMINI_AUTHORIZATION_HEADER] = gemini_api_key

    request_body_bytes = await request.body()
    request_body_json = json.loads(request_body_bytes)
    print("Here is the request: ", request_body_json)

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
    return await handle_non_streaming_response(
        response, dataset_name, request_body_json, invariant_authorization
    )


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

            # Yield chunk immediately to the client
            print("Here is the response chunk: ", chunk)
            yield chunk

        # Send full merged response to the explorer
        if dataset_name:
            # Push to Explorer
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def push_to_explorer(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    converted_requests = convert_request(request_body)
    converted_responses = convert_response(merged_response)
    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[converted_requests + converted_responses],
        invariant_authorization=invariant_authorization,
    )


async def handle_non_streaming_response(
    response: httpx.Response,
    dataset_name: Optional[str],
    request_body_json: dict[str, Any],
    invariant_authorization: Optional[str],
) -> Response:
    """Handles non-streaming Gemini responses"""
    try:
        json_response = response.json()
        print("Here is the response: ", json_response)
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
        await push_to_explorer(
            dataset_name, json_response, request_body_json, invariant_authorization
        )

    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

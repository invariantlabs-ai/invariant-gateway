"""Gateway service to forward requests to the Gemini APIs"""

import asyncio
import json
from typing import Any

import httpx
from common.config_manager import GatewayConfig, GatewayConfigManager
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from common.authorization import extract_authorization_from_headers
from common.request_context_data import RequestContextData
from converters.gemini_to_invariant import convert_request, convert_response
from integrations.explorer import push_trace

gateway = APIRouter()

GEMINI_AUTHORIZATION_HEADER = "x-goog-api-key"


@gateway.post("/gemini/{api_version}/models/{model}:{endpoint}")
@gateway.post("/{dataset_name}/gemini/{api_version}/models/{model}:{endpoint}")
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
    request_json = json.loads(request_body_bytes)

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

    context = RequestContextData(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
    )

    if alt == "sse" or endpoint == "streamGenerateContent":
        return await stream_response(
            context,
            client,
            gemini_request,
        )
    response = await client.send(gemini_request)
    return await handle_non_streaming_response(
        context,
        response,
    )


async def stream_response(
    context: RequestContextData,
    client: httpx.AsyncClient,
    gemini_request: httpx.Request,
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
        # Store the progressively merged response
        merged_response = {
            "candidates": [{"content": {"parts": []}, "finishReason": None}]
        }

        async for chunk in response.aiter_bytes():
            chunk_text = chunk.decode().strip()
            if not chunk_text:
                continue

            # Yield chunk immediately to the client
            yield chunk

            # Parse and update merged_response incrementally
            process_chunk_text(merged_response, chunk_text)

        if context.dataset_name:
            # Push to Explorer - don't block on the response
            asyncio.create_task(
                push_to_explorer(
                    context,
                    merged_response,
                )
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def process_chunk_text(
    merged_response: dict[str, Any],
    chunk_text: str,
) -> None:
    """Processes the chunk text and updates the merged_response to be sent to the explorer"""
    # Split the chunk text into individual JSON strings
    # A single chunk can contain multiple "data: " sections
    for json_string in chunk_text.split("data: "):
        json_string = json_string.replace("data: ", "").strip()

        if not json_string:
            continue

        try:
            json_chunk = json.loads(json_string)
        except json.JSONDecodeError:
            print("Warning: Could not parse chunk:", json_string)

        update_merged_response(merged_response, json_chunk)


def update_merged_response(merged_response: dict[str, Any], chunk_json: dict) -> None:
    """Updates the merged response incrementally with a new chunk."""
    candidates = chunk_json.get("candidates", [])

    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "text" in part:
                existing_parts = merged_response["candidates"][0]["content"]["parts"]
                if existing_parts and "text" in existing_parts[-1]:
                    existing_parts[-1]["text"] += part["text"]
                else:
                    existing_parts.append({"text": part["text"]})

            if "functionCall" in part:
                merged_response["candidates"][0]["content"]["parts"].append(
                    {"functionCall": part["functionCall"]}
                )

        if "role" in content:
            merged_response["candidates"][0]["content"]["role"] = content["role"]

        if "finishReason" in candidate:
            merged_response["candidates"][0]["finishReason"] = candidate["finishReason"]


async def push_to_explorer(
    context: RequestContextData,
    response_json: dict[str, Any],
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    converted_requests = convert_request(context.request_json)
    converted_responses = convert_response(response_json)
    _ = await push_trace(
        dataset_name=context.dataset_name,
        messages=[converted_requests + converted_responses],
        invariant_authorization=context.invariant_authorization,
    )


async def handle_non_streaming_response(
    context: RequestContextData,
    response: httpx.Response,
) -> Response:
    """Handles non-streaming Gemini responses"""
    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail="Invalid JSON response received from Gemini API",
        ) from e
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response_json.get("error", "Unknown error from Gemini API"),
        )
    if context.dataset_name:
        # Push to Explorer - don't block on the response
        asyncio.create_task(push_to_explorer(context, response_json))

    return Response(
        content=json.dumps(response_json),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

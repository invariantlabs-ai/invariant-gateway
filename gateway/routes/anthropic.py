"""Gateway service to forward requests to the Anthropic APIs"""

import asyncio
import json
from typing import Any

import httpx
from common.config_manager import GatewayConfig, GatewayConfigManager
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from integrations.explorer import push_trace
from converters.anthropic_to_invariant import (
    convert_anthropic_to_invariant_message_format,
)
from common.authorization import extract_authorization_from_headers
from common.request_context_data import RequestContextData

gateway = APIRouter()

MISSING_ANTHROPIC_AUTH_HEADER = "Missing Anthropic authorization header"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "
END_REASONS = ["end_turn", "max_tokens", "stop_sequence"]

MESSAGE_START = "message_start"
MESSGAE_DELTA = "message_delta"
MESSAGE_STOP = "message_stop"
CONTENT_BLOCK_START = "content_block_start"
CONTENT_BLOCK_DELTA = "content_block_delta"
CONTENT_BLOCK_STOP = "content_block_stop"

ANTHROPIC_AUTHORIZATION_HEADER = "x-api-key"


def validate_headers(x_api_key: str = Header(None)):
    """Require the headers to be present"""
    if x_api_key is None:
        raise HTTPException(status_code=400, detail=MISSING_ANTHROPIC_AUTH_HEADER)


@gateway.post(
    "/{dataset_name}/anthropic/v1/messages",
    dependencies=[Depends(validate_headers)],
)
@gateway.post(
    "/anthropic/v1/messages",
    dependencies=[Depends(validate_headers)],
)
async def anthropic_v1_messages_gateway(
    request: Request,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),  # pylint: disable=unused-argument
):
    """Proxy calls to the Anthropic APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, anthopic_api_key = extract_authorization_from_headers(
        request, dataset_name, ANTHROPIC_AUTHORIZATION_HEADER
    )
    headers[ANTHROPIC_AUTHORIZATION_HEADER] = anthopic_api_key

    request_body = await request.body()
    request_json = json.loads(request_body)
    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    anthropic_request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        data=request_body,
    )

    context = RequestContextData(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
    )

    if request_json.get("stream"):
        return await handle_streaming_response(context, client, anthropic_request)
    response = await client.send(anthropic_request)
    return await handle_non_streaming_response(context, response)


async def push_to_explorer(
    context: RequestContextData,
    merged_response: dict[str, Any],
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    # Combine the messages from the request body and Anthropic response
    messages = context.request_json.get("messages", [])
    messages += [merged_response]

    converted_messages = convert_anthropic_to_invariant_message_format(messages)
    _ = await push_trace(
        dataset_name=context.dataset_name,
        messages=[converted_messages],
        invariant_authorization=context.invariant_authorization,
    )


async def handle_non_streaming_response(
    context: RequestContextData,
    response: httpx.Response,
) -> Response:
    """Handles non-streaming Anthropic responses"""
    try:
        json_response = response.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Invalid JSON response received from Anthropic: {response.text}, got error{e}",
        ) from e
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=json_response.get("error", "Unknown error from Anthropic"),
        )
    # Only push the trace to explorer if the last message is an end turn message
    # Don't block on the response from explorer
    if context.dataset_name:
        asyncio.create_task(push_to_explorer(context, json_response))
    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )


async def handle_streaming_response(
    context: RequestContextData,
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
) -> StreamingResponse:
    """Handles streaming Anthropic responses"""
    merged_response = []

    response = await client.send(anthropic_request, stream=True)
    if response.status_code != 200:
        error_content = await response.aread()
        try:
            error_json = json.loads(error_content)
            error_detail = error_json.get("error", "Unknown error from Anthropic")
        except json.JSONDecodeError:
            error_detail = {"error": "Failed to decode error response from Anthropic"}
        raise HTTPException(status_code=response.status_code, detail=error_detail)

    async def event_generator() -> Any:
        async for chunk in response.aiter_bytes():
            chunk_decode = chunk.decode().strip()
            if not chunk_decode:
                continue
            yield chunk

            process_chunk_text(chunk_decode, merged_response)
        if context.dataset_name:
            # Push to Explorer - don't block on the response
            asyncio.create_task(push_to_explorer(context, merged_response[-1]))

    generator = event_generator()

    return StreamingResponse(generator, media_type="text/event-stream")


def process_chunk_text(chunk_decode, merged_response):
    """
    Process the chunk of text and update the merged_response
    Example of chunk list can be find in:
    ../../resources/streaming_chunk_text/anthropic.txt
    """
    for text_block in chunk_decode.split("\n\n"):
        # might be empty block
        if len(text_block.split("\ndata:")) > 1:
            text_data = text_block.split("\ndata:")[1]
            text_json = json.loads(text_data)
            update_merged_response(text_json, merged_response)


def update_merged_response(text_json, merged_response):
    """Update the formatted_invariant_response based on the text_json"""
    if text_json.get("type") == MESSAGE_START:
        message = text_json.get("message")
        merged_response.append(
            {
                "id": message.get("id"),
                "role": message.get("role"),
                "content": "",
                "model": message.get("model"),
                "stop_reason": message.get("stop_reason"),
                "stop_sequence": message.get("stop_sequence"),
            }
        )
    elif (
        text_json.get("type") == CONTENT_BLOCK_START
        and text_json.get("content_block").get("type") == "tool_use"
    ):
        content_block = text_json.get("content_block")
        merged_response.append(
            {
                "role": "tool",
                "tool_id": content_block.get("id"),
                "content": "",
            }
        )
    elif text_json.get("type") == CONTENT_BLOCK_DELTA:
        if merged_response[-1]["role"] == "assistant":
            merged_response[-1]["content"] += text_json.get("delta").get("text")
        elif merged_response[-1]["role"] == "tool":
            merged_response[-1]["content"] += text_json.get("delta").get("partial_json")
    elif text_json.get("type") == MESSGAE_DELTA:
        merged_response[-1]["stop_reason"] = text_json.get("delta").get("stop_reason")

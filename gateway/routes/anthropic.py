"""Gateway service to forward requests to the Anthropic APIs"""

import asyncio
import json
from typing import Any, Optional

import httpx
from common.config_manager import GatewayConfig, GatewayConfigManager
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from integrations.explorer import create_annotations_from_guardrails_errors, push_trace
from converters.anthropic_to_invariant import (
    convert_anthropic_to_invariant_message_format,
)
from common.authorization import extract_authorization_from_headers
from common.request_context_data import RequestContextData
from integrations.guardails import check_guardrails, preload_guardrails

gateway = APIRouter()

MISSING_ANTHROPIC_AUTH_HEADER = "Missing Anthropic authorization header"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "
END_REASONS = ["end_turn", "max_tokens", "stop_sequence"]

MESSAGE_START = "message_start"
MESSAGE_DELTA = "message_delta"
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
        config=config,
    )
    asyncio.create_task(preload_guardrails(context))

    if request_json.get("stream"):
        return await handle_streaming_response(context, client, anthropic_request)
    response = await client.send(anthropic_request)
    return await handle_non_streaming_response(context, response)


def create_metadata(
    context: RequestContextData, response_json: dict[str, Any]
) -> dict[str, Any]:
    """Creates metadata for the trace"""
    print("[DEBUG] Anthropic Request JSON: ", context.request_json, flush=True)
    print("[DEBUG] Anthropic Response JSON: ", response_json, flush=True)
    metadata = {k: v for k, v in context.request_json.items() if k != "messages"}
    metadata["via_gateway"] = True
    if response_json.get("usage"):
        metadata["usage"] = response_json.get("usage")
    print("[DEBUG] Anthropic Metadata: ", metadata, flush=True)
    return metadata


def combine_request_and_response_messages(
    context: RequestContextData, json_response: dict[str, Any]
):
    """Combine the request and response messages"""
    messages = []
    if "system" in context.request_json:
        messages.append(
            {"role": "system", "content": context.request_json.get("system")}
        )
    messages.extend(context.request_json.get("messages", []))
    messages.append(json_response)
    return messages


async def get_guardrails_check_result(
    context: RequestContextData, json_response: dict[str, Any]
) -> dict[str, Any]:
    """Get the guardrails check result"""
    messages = combine_request_and_response_messages(context, json_response)
    converted_messages = convert_anthropic_to_invariant_message_format(messages)

    # Block on the guardrails check
    guardrails_execution_result = await check_guardrails(
        messages=converted_messages,
        guardrails=context.config.guardrails,
        invariant_authorization=context.invariant_authorization,
    )
    return guardrails_execution_result


async def push_to_explorer(
    context: RequestContextData,
    merged_response: dict[str, Any],
    guardrails_execution_result: Optional[dict] = None,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    guardrails_execution_result = guardrails_execution_result or {}
    annotations = create_annotations_from_guardrails_errors(
        guardrails_execution_result.get("errors", [])
    )

    # Combine the messages from the request body and Anthropic response
    messages = combine_request_and_response_messages(context, merged_response)

    converted_messages = convert_anthropic_to_invariant_message_format(messages)
    _ = await push_trace(
        dataset_name=context.dataset_name,
        messages=[converted_messages],
        invariant_authorization=context.invariant_authorization,
        metadata=[create_metadata(context, merged_response)],
        annotations=[annotations] if annotations else None,
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

    guardrails_execution_result = {}
    response_string = json.dumps(json_response)
    response_code = response.status_code

    if context.config and context.config.guardrails:
        # Block on the guardrails check
        guardrails_execution_result = await get_guardrails_check_result(
            context, json_response
        )
        if guardrails_execution_result.get("errors", []):
            response_string = json.dumps(
                {
                    "error": "[Invariant] The response did not pass the guardrails",
                    "details": guardrails_execution_result,
                }
            )
            response_code = 400
    if context.dataset_name:
        # Push to Explorer - don't block on its response
        asyncio.create_task(
            push_to_explorer(context, json_response, guardrails_execution_result)
        )

    updated_headers = response.headers.copy()
    updated_headers.pop("Content-Length", None)
    return Response(
        content=response_string,
        status_code=response_code,
        media_type="application/json",
        headers=dict(updated_headers),
    )


async def handle_streaming_response(
    context: RequestContextData,
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
) -> StreamingResponse:
    """Handles streaming Anthropic responses"""
    merged_response = {}

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
            decoded_chunk = chunk.decode().strip()
            if not decoded_chunk:
                continue
            process_chunk(decoded_chunk, merged_response)
            if (
                "event: message_stop" in decoded_chunk
                and context.config
                and context.config.guardrails
            ):
                # Block on the guardrails check
                guardrails_execution_result = await get_guardrails_check_result(
                    context, merged_response
                )
                if guardrails_execution_result.get("errors", []):
                    error_chunk = json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "message": "[Invariant] The response did not pass the guardrails",
                                "details": guardrails_execution_result,
                            },
                        }
                    )
                    # Push annotated trace to the explorer - don't block on its response
                    if context.dataset_name:
                        asyncio.create_task(
                            push_to_explorer(
                                context,
                                merged_response,
                                guardrails_execution_result,
                            )
                        )
                    yield f"event: error\ndata: {error_chunk}\n\n".encode()
                    return
            yield chunk

        if context.dataset_name:
            # Push to Explorer - don't block on the response
            asyncio.create_task(push_to_explorer(context, merged_response))

    generator = event_generator()

    return StreamingResponse(generator, media_type="text/event-stream")


def process_chunk(chunk: str, merged_response: dict[str, Any]) -> None:
    """
    Process the chunk of text and update the merged_response
    Example of chunk list can be find in:
    ../../resources/streaming_chunk_text/anthropic.txt
    """
    for text_block in chunk.split("\n\n"):
        # might be empty block
        if len(text_block.split("\ndata:")) > 1:
            event_text = text_block.split("\ndata:")[1]
            event = json.loads(event_text)
            update_merged_response(event, merged_response)


def update_merged_response(
    event: dict[str, Any], merged_response: dict[str, Any]
) -> None:
    """
    Update the merged_response based on the event.

    Each stream uses the following event flow:

    1. message_start: contains a Message object with empty content.
    2. A series of content blocks, each of which have a content_block_start,
    one or more content_block_delta events, and a content_block_stop event.
    Each content block will have an index that corresponds to its index in the
    final Message content array.
    3. One or more message_delta events, indicating top-level changes to the final Message object.
    A final message_stop event.

    """
    if event.get("type") == MESSAGE_START:
        merged_response.update(**event.get("message"))
    elif event.get("type") == CONTENT_BLOCK_START:
        index = event.get("index")
        if index >= len(merged_response.get("content")):
            merged_response["content"].append(event.get("content_block"))
        if event.get("content_block").get("type") == "tool_use":
            merged_response.get("content")[-1]["input"] = ""
    elif event.get("type") == CONTENT_BLOCK_DELTA:
        index = event.get("index")
        delta = event.get("delta")
        if delta.get("type") == "text_delta":
            merged_response.get("content")[index]["text"] += delta.get("text")
        elif delta.get("type") == "input_json_delta":
            merged_response.get("content")[index]["input"] += delta.get("partial_json")
    elif event.get("type") == MESSAGE_DELTA:
        merged_response["usage"].update(**event.get("usage"))

"""Proxy service to forward requests to the Anthropic APIs"""

import json
from typing import Any, Optional

import httpx
from common.config_manager import ProxyConfig, ProxyConfigManager
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from utils.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
    INVARIANT_AUTHORIZATION_HEADER,
)
from utils.explorer import push_trace

proxy = APIRouter()

MISSING_INVARIANT_AUTH_API_KEY = "Missing invariant authorization header"
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


@proxy.post(
    "/{dataset_name}/anthropic/v1/messages",
    dependencies=[Depends(validate_headers)],
)
@proxy.post(
    "/anthropic/v1/messages",
    dependencies=[Depends(validate_headers)],
)
async def anthropic_v1_messages_proxy(
    request: Request,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
    config: ProxyConfig = Depends(ProxyConfigManager.get_config),  # pylint: disable=unused-argument
):
    """Proxy calls to the Anthropic APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    # In case the user wants to push to Explorer, the request must contain the Invariant API Key
    # The invariant-authorization header contains the Invariant API Key
    # "invariant-authorization": "Bearer <Invariant API Key>"
    # The x-api-key header contains the Anthropic API Key
    # "x-api-key": "<Anthropic API Key>"
    #
    # For some clients, it is not possible to pass a custom header
    # In such cases, the Invariant API Key is passed as part of the
    # x-api-key header with the Anthropic API key.
    # The header in that case becomes:
    # "x-api-key": "<Anthropic API Key>|invariant-auth: <Invariant API Key>"
    invariant_authorization = None
    if dataset_name:
        if request.headers.get(
            INVARIANT_AUTHORIZATION_HEADER
        ) is None and "|invariant-auth:" not in request.headers.get(
            ANTHROPIC_AUTHORIZATION_HEADER
        ):
            raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_API_KEY)
        if request.headers.get(INVARIANT_AUTHORIZATION_HEADER):
            invariant_authorization = request.headers.get(
                INVARIANT_AUTHORIZATION_HEADER
            )
        else:
            header_value = request.headers.get(ANTHROPIC_AUTHORIZATION_HEADER)
            api_keys = header_value.split("|invariant-auth: ")
            invariant_authorization = f"Bearer {api_keys[1].strip()}"
            # Update the authorization header to pass the Anthropic API Key
            headers[ANTHROPIC_AUTHORIZATION_HEADER] = f"{api_keys[0].strip()}"

    request_body = await request.body()

    request_body_json = json.loads(request_body)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))

    anthropic_request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        data=request_body,
    )

    if request_body_json.get("stream"):
        return await handle_streaming_response(
            client, anthropic_request, dataset_name, invariant_authorization
        )
    response = await client.send(anthropic_request)
    return await handle_non_streaming_response(
        response, dataset_name, request_body_json, invariant_authorization
    )


async def push_to_explorer(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    # Combine the messages from the request body and Anthropic response
    messages = request_body.get("messages", [])
    messages += [merged_response]

    transformed_messages = convert_anthropic_to_invariant_message_format(messages)
    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[transformed_messages],
        invariant_authorization=invariant_authorization,
    )


async def handle_non_streaming_response(
    response: httpx.Response,
    dataset_name: Optional[str],
    request_body_json: dict[str, Any],
    invariant_authorization: Optional[str],
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
    if dataset_name:
        await push_to_explorer(
            dataset_name,
            json_response,
            request_body_json,
            invariant_authorization,
        )
    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )


async def handle_streaming_response(
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
    dataset_name: Optional[str],
    invariant_authorization: Optional[str],
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
        if dataset_name:
            await push_to_explorer(
                dataset_name,
                merged_response[-1],
                json.loads(anthropic_request.content),
                invariant_authorization,
            )

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


def convert_anthropic_to_invariant_message_format(
    messages: list[dict], keep_empty_tool_response: bool = False
) -> list[dict]:
    """Converts a list of messages from the Anthropic API to the Invariant API format."""
    output = []
    role_mapping = {
        "system": lambda msg: {"role": "system", "content": msg["content"]},
        "user": lambda msg: handle_user_message(msg, keep_empty_tool_response),
        "assistant": lambda msg: handle_assistant_message(msg),
    }

    for message in messages:
        handler = role_mapping.get(message["role"])
        if handler:
            output.extend(handler(message))

    return output


def handle_user_message(message, keep_empty_tool_response):
    """Handle the user message from the Anthropic API"""
    output = []
    content = message["content"]
    if isinstance(content, list):
        user_content = []
        for sub_message in content:
            if sub_message["type"] == "tool_result":
                if sub_message["content"]:
                    output.append(
                        {
                            "role": "tool",
                            "content": sub_message["content"],
                            "tool_id": sub_message["tool_use_id"],
                        }
                    )
                elif keep_empty_tool_response and any(sub_message.values()):
                    output.append(
                        {
                            "role": "tool",
                            "content": {"is_error": True}
                            if sub_message["is_error"]
                            else {},
                            "tool_id": sub_message["tool_use_id"],
                        }
                    )
            elif sub_message["type"] == "text":
                user_content.append({"type": "text", "text": sub_message["text"]})
            elif sub_message["type"] == "image":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:"
                            + sub_message["source"]["media_type"]
                            + ";base64,"
                            + sub_message["source"]["data"],
                        },
                    },
                )
        if user_content:
            output.append({"role": "user", "content": user_content})
    else:
        output.append({"role": "user", "content": content})
    return output


def handle_assistant_message(message):
    """Handle the assistant message from the Anthropic API"""
    output = []
    if isinstance(message["content"], list):
        for sub_message in message["content"]:
            if sub_message["type"] == "text":
                output.append({"role": "assistant", "content": sub_message.get("text")})
            elif sub_message["type"] == "tool_use":
                output.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "tool_id": sub_message.get("id"),
                                "type": "function",
                                "function": {
                                    "name": sub_message.get("name"),
                                    "arguments": sub_message.get("input"),
                                },
                            }
                        ],
                    }
                )
    else:
        output.append({"role": "assistant", "content": message["content"]})
    return output

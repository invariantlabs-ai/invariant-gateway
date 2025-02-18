"""Proxy service to forward requests to the Anthropic APIs"""

import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from utils.constants import CLIENT_TIMEOUT, IGNORED_HEADERS
from utils.explorer import push_trace
from starlette.responses import StreamingResponse


proxy = APIRouter()

ALLOWED_ANTHROPIC_ENDPOINTS = {"v1/messages"}

MISSING_INVARIANT_AUTH_HEADER = "Missing invariant-authorization header"
MISSING_ANTHROPIC_AUTH_HEADER = "Missing athropic authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported Anthropic endpoint"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "
END_REASONS = ["end_turn", "max_tokens", "stop_sequence"]


MESSAGE_START = "message_start"
MESSGAE_DELTA = "message_delta"
MESSAGE_STOP = "message_stop"
CONTENT_BLOCK_START = "content_block_start"
CONTENT_BLOCK_DELTA = "content_block_delta"
CONTENT_BLOCK_STOP = "content_block_stop"

def validate_headers(
    invariant_authorization: str = Header(None), x_api_key: str = Header(None)
):
    """Require the invariant-authorization and authorization headers to be present"""
    if invariant_authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_HEADER)
    if x_api_key is None:
        raise HTTPException(status_code=400, detail=MISSING_ANTHROPIC_AUTH_HEADER)


@proxy.post(
    "/{dataset_name}/anthropic/{endpoint:path}",
    dependencies=[Depends(validate_headers)],
)
async def anthropic_proxy(
    dataset_name: str,
    endpoint: str,
    request: Request,
):
    """Proxy calls to the Anthropic APIs"""
    if endpoint not in ALLOWED_ANTHROPIC_ENDPOINTS:
        raise HTTPException(status_code=404, detail=NOT_SUPPORTED_ENDPOINT)
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    request_body = await request.body()

    request_body_json = json.loads(request_body)

    anthropic_url = f"https://api.anthropic.com/{endpoint}"

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))

    anthropic_request = client.build_request(
        "POST", anthropic_url, headers=headers, data=request_body
    )
    invariant_authorization = request.headers.get("invariant-authorization")

    if request_body_json.get("stream"):
        return await handle_streaming_response(client, anthropic_request, dataset_name, invariant_authorization)
    else:
        try:    
            response = await client.send(anthropic_request)
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch response: {response.text}, got error{e}",
            )
        await handle_non_streaming_response(
            response, dataset_name, request_body_json, invariant_authorization
        )
        return response.json()


async def push_to_explorer(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
    reformat: bool = True,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    # Combine the messages from the request body and Anthropic response
    messages = request_body.get("messages", [])
    messages += [merged_response]

    if reformat:
        messages = anthropic_to_invariant_messages(messages)
    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[messages],
        invariant_authorization=invariant_authorization,
    )


async def handle_non_streaming_response(
    response: httpx.Response,
    dataset_name: str,
    request_body_json: dict[str, Any],
    invariant_authorization: str,
):
    """Handles non-streaming Anthropic responses"""
    json_response = response.json()
    # Only push the trace to explorer if the last message is an end turn message
    if json_response.get("stop_reason") in END_REASONS:
        await push_to_explorer(
            dataset_name,
            json_response,
            request_body_json,
            invariant_authorization,
        )

async def handle_streaming_response(
        client: httpx.AsyncClient,
        anthropic_request: httpx.Request,
        dataset_name: str,
        invariant_authorization: str
) -> StreamingResponse:

    formatted_invariant_response = []
    
    async def event_generator() -> Any:
        async with client.stream(
            "POST",
            anthropic_request.url,
            headers=anthropic_request.headers,
            content=anthropic_request.content,
        ) as response:
            if response.status_code != 200:
                yield json.dumps(
                    {"error": f"Failed to fetch response: {response.status_code}"}
                ).encode()
                return
            async for chunk in response.aiter_bytes():
                yield chunk

                process_chunk_text(
                    chunk,
                    formatted_invariant_response
                )

            if formatted_invariant_response and formatted_invariant_response[-1].get("stop_reason") in END_REASONS:
                await push_to_explorer(
                    dataset_name,
                    formatted_invariant_response[-1],
                    json.loads(anthropic_request.content),
                    invariant_authorization,
                )
    
    generator = event_generator()
        
    return StreamingResponse(generator, media_type="text/event-stream")


def process_chunk_text(chunk, formatted_invariant_response):
    """
    Process the chunk of text and update the formatted_invariant_response
    Example of chunk list can be find in:
    ../../resources/streaming_chunk_text/anthropic.txt
    """
    text_decode = chunk.decode().strip()
    for text_block in text_decode.split("\n\n"):
        # might be empty block
      
        if len(text_block.split("\ndata:"))>1:
            text_data = text_block.split("\ndata:")[1]
            text_json = json.loads(text_data)
            update_formatted_invariant_response(text_json, formatted_invariant_response)

def update_formatted_invariant_response(text_json, formatted_invariant_response):
    if text_json.get("type") == MESSAGE_START:
        message = text_json.get("message")
        formatted_invariant_response.append({
            "id": message.get("id"),
            "role": message.get("role"),
            "content": "",
            "model": message.get("model"),
            "stop_reason": message.get("stop_reason"),
            "stop_sequence": message.get("stop_sequence"),
        })
    elif text_json.get("type") == CONTENT_BLOCK_START and text_json.get("content_block").get("type")=="tool_use":
        content_block = text_json.get("content_block")
        formatted_invariant_response.append(
            {
                "role": "tool",
                "tool_id": content_block.get("id"),
                "content": "",
            }
        )
    elif text_json.get("type") == CONTENT_BLOCK_DELTA:
        if formatted_invariant_response[-1]["role"]=="assistant":
            formatted_invariant_response[-1]["content"] += text_json.get("delta").get("text")
        elif formatted_invariant_response[-1]["role"]=="tool":
            formatted_invariant_response[-1]["content"] += text_json.get("delta").get("partial_json")
    elif text_json.get("type") == MESSGAE_DELTA:
        formatted_invariant_response[-1]["stop_reason"] = text_json.get("delta").get("stop_reason")

def anthropic_to_invariant_messages(
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
    output = []
    content = message["content"]
    if isinstance(content, list):
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
                output.append({"role": "user", "content": sub_message["text"]})
    else:
        output.append({"role": "user", "content": content})
    return output


def handle_assistant_message(message):
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

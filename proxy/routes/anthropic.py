"""Proxy service to forward requests to the Anthropic APIs"""

from fastapi import APIRouter, Header, HTTPException, Depends, Request
import json
import httpx
from typing import Any
from utils.explorer import push_trace
# from .open_ai import push_to_explorer

proxy = APIRouter()

ALLOWED_ANTHROPIC_ENDPOINTS = {"v1/messages"}
IGNORED_HEADERS = [
    "accept-encoding",
    "host",
    "invariant-authorization",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
    "x-real-ip",
]

MISSING_INVARIANT_AUTH_HEADER = "Missing invariant-authorization header"
MISSING_AUTH_HEADER = "Missing authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported OpenAI endpoint"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "
END_REASONS = [
    "end_turn",
    "max_tokens",
    "stop_sequence"
]

def validate_headers(
        invariant_authorization: str = Header(None), authorization: str = Header(None)
):
    """Require the invariant-authorization and authorization headers to be present"""
    if invariant_authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_HEADER)
    # if authorization is None:
    #     raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)
    
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
    # headers["accept-encoding"] = "identity"

    request_body = await request.body()

    request_body_json = json.loads(request_body)

    anthropic_url = f"https://api.anthropic.com/{endpoint}"
    client = httpx.AsyncClient()

    anthropic_request = client.build_request(
        "POST", 
        anthropic_url, 
        headers=headers, 
        data=request_body
    )

    invariant_authorization = request.headers.get("invariant-authorization")

    async with client:
        response = await client.send(anthropic_request)
        await handle_non_streaming_response(
            response, dataset_name, request_body_json, invariant_authorization
        )
        return response.json()

async def push_to_explorer(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    # Combine the messages from the request body and the choices from the OpenAI response
    messages = request_body.get("messages", [])
    if merged_response is not list:
        merged_response = [merged_response]
    messages += merged_response
    if messages[-1].get("stop_reason") in END_REASONS:
        messages = anthropic_to_invariant_messages(messages)
        response = await push_trace(
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
    await push_to_explorer(
        dataset_name,
        json_response,
        request_body_json,
        invariant_authorization,
    )

def anthropic_to_invariant_messages(
    messages: list[dict], keep_empty_tool_response: bool = False
) -> list[dict]:
    """Converts a list of messages from the Anthropic API to the Invariant API format."""
    output = []
    for message in messages:
        if message["role"] == "system":
            output.append({"role": "system", "content": message["content"]})
        if message["role"] == "user":
            if isinstance(message["content"], list):
                for sub_message in message["content"]:
                    if sub_message["type"] == "tool_result":
                        if sub_message["content"]:
                            output.append(
                                {
                                    "role": "tool",
                                    "content": sub_message["content"],
                                    "tool_id": sub_message["tool_use_id"],
                                }
                            )
                        else:
                            if keep_empty_tool_response and any(
                                [sub_message[k] for k in sub_message]
                            ):
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
                output.append({"role": "user", "content": message["content"]})
        if message["role"] == "assistant":
            for sub_message in message["content"]:
                if sub_message["type"] == "text":
                    output.append(
                        {"role": "assistant", "content": sub_message.get("text")}
                    )
                if sub_message["type"] == "tool_use":
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
    return output
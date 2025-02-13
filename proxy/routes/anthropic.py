"""Proxy service to forward requests to the Anthropic APIs"""

import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from utils.constants import IGNORED_HEADERS
from utils.explorer import push_trace

proxy = APIRouter()

ALLOWED_ANTHROPIC_ENDPOINTS = {"v1/messages"}

MISSING_INVARIANT_AUTH_HEADER = "Missing invariant-authorization header"
MISSING_ANTHROPIC_AUTH_HEADER = "Missing athropic authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported OpenAI endpoint"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "
END_REASONS = ["end_turn", "max_tokens", "stop_sequence"]


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

    request_body = await request.body()

    request_body_json = json.loads(request_body)

    anthropic_url = f"https://api.anthropic.com/{endpoint}"
    client = httpx.AsyncClient()

    anthropic_request = client.build_request(
        "POST", anthropic_url, headers=headers, data=request_body
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
    # Combine the messages from the request body and Anthropic response
    messages = request_body.get("messages", [])
    messages += [merged_response]

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
    return output

"""Proxy service to forward requests to the OpenAI APIs"""

import json

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from utils.explorer import push_trace

ALLOWED_OPEN_AI_ENDPOINTS = {"chat/completions"}
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

proxy = APIRouter()

MISSING_INVARIANT_AUTH_HEADER = "Missing invariant-authorization header"
MISSING_AUTH_HEADER = "Missing authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported OpenAI endpoint"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "


def validate_headers(
    invariant_authorization: str = Header(None), authorization: str = Header(None)
):
    """Require the invariant-authorization and authorization headers to be present"""
    if invariant_authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_HEADER)
    if authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)


@proxy.post(
    "/{dataset_name}/openai/{endpoint:path}",
    dependencies=[Depends(validate_headers)],
)
async def openai_proxy(
    request: Request,
    dataset_name: str,
    endpoint: str,
):
    """Proxy calls to the OpenAI APIs"""
    if endpoint not in ALLOWED_OPEN_AI_ENDPOINTS:
        raise HTTPException(status_code=404, detail=NOT_SUPPORTED_ENDPOINT)

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    request_body_bytes = await request.body()
    request_body_json = json.loads(request_body_bytes)

    # Check if the request is for streaming
    is_streaming = request_body_json.get("stream", False)

    client = httpx.AsyncClient()
    open_ai_request = client.build_request(
        "POST",
        f"https://api.openai.com/v1/{endpoint}",
        content=request_body_bytes,
        headers=headers,
    )
    if is_streaming:
        return await stream_response(
            client,
            open_ai_request,
            dataset_name,
            request_body_json,
            request.headers,
        )
    else:
        async with client:
            response = await client.send(open_ai_request)
            return await handle_non_streaming_response(
                response, dataset_name, request_body_json, request.headers
            )


async def stream_response(
    client, open_ai_request, dataset_name, request_body_json, request_headers
):
    """Handles streaming the OpenAI response to the client while collecting full response"""

    async def event_generator():
        full_response = {
            "id": None,
            "object": "chat.completion",
            "created": None,
            "model": None,
            "choices": [],
            "usage": None,
        }

        # Tracks choice index to full_response index
        index_mapping = {}
        # Tracks tool calls by index
        tool_call_mapping = {}

        async with client.stream(
            "POST",
            open_ai_request.url,
            headers=open_ai_request.headers,
            content=open_ai_request.content,
        ) as response:
            if response.status_code != 200:
                error_message = json.dumps(
                    {"error": f"Failed to fetch response: {response.status_code}"}
                ).encode()
                yield error_message
                return

            async for chunk in response.aiter_bytes():
                chunk_text = chunk.decode().strip()
                if not chunk_text:
                    continue

                # Yield chunk immediately to the client (proxy behavior)
                yield chunk

                # There can be multiple "data: " chunks in a single response
                for json_string in chunk_text.split("\ndata: "):
                    # Remove first "data: " prefix
                    json_string = json_string.replace("data: ", "").strip()

                    if not json_string or json_string == "[DONE]":
                        continue

                    try:
                        json_chunk = json.loads(json_string)
                    except json.JSONDecodeError:
                        continue

                    # Extract metadata safely
                    full_response["id"] = full_response["id"] or json_chunk.get("id")
                    full_response["created"] = full_response[
                        "created"
                    ] or json_chunk.get("created")
                    full_response["model"] = full_response["model"] or json_chunk.get(
                        "model"
                    )

                    for choice in json_chunk.get("choices", []):
                        index = choice.get("index", 0)

                        # Ensure we have a mapping for this index
                        if index not in index_mapping:
                            index_mapping[index] = len(full_response["choices"])
                            full_response["choices"].append(
                                {
                                    "index": index,
                                    "message": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            )

                        existing_choice = full_response["choices"][index_mapping[index]]
                        delta = choice.get("delta", {})

                        # Handle regular assistant messages
                        content = delta.get("content")
                        if content is not None:
                            if "content" not in existing_choice["message"]:
                                existing_choice["message"]["content"] = ""
                            existing_choice["message"]["content"] += content

                        # Handle tool calls
                        if isinstance(delta.get("tool_calls"), list):
                            if "tool_calls" not in existing_choice["message"]:
                                existing_choice["message"]["tool_calls"] = []

                            for tool in delta["tool_calls"]:
                                tool_index = tool.get("index")
                                tool_id = tool.get("id")
                                tool_name = tool.get("function", {}).get("name")
                                tool_arguments = tool.get("function", {}).get(
                                    "arguments", ""
                                )

                                if tool_index is None:
                                    continue

                                # Find or create tool call by index
                                if tool_index not in tool_call_mapping:
                                    tool_call_mapping[tool_index] = {
                                        "index": tool_index,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": "",
                                        },
                                    }
                                    existing_choice["message"]["tool_calls"].append(
                                        tool_call_mapping[tool_index]
                                    )

                                tool_entry = tool_call_mapping[tool_index]

                                if tool_id:
                                    tool_entry["id"] = tool_id

                                if tool_name:
                                    tool_entry["function"]["name"] = tool_name

                                # Append arguments if they exist
                                if tool_arguments:
                                    tool_entry["function"]["arguments"] += (
                                        tool_arguments
                                    )

                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            existing_choice["finish_reason"] = finish_reason

            # Send full merged response to the explorer
            await push_to_explorer(
                dataset_name, full_response, request_headers, request_body_json
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def push_to_explorer(dataset_name, full_response, request_headers, request_body):
    """Pushes the full trace to the Invariant Explorer"""
    # Combine messages from the request and the response
    # to push the full trace to the Invariant Explorer
    messages = request_body.get("messages", [])
    messages += [choice["message"] for choice in full_response.get("choices", [])]

    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[messages],
        invariant_authorization=request_headers.get("invariant-authorization"),
    )


async def handle_non_streaming_response(
    response, dataset_name, request_body_json, request_headers
):
    """Handles non-streaming OpenAI responses"""
    json_response = response.json()
    await push_to_explorer(
        dataset_name, json_response, request_headers, request_body_json
    )

    response_headers = dict(response.headers)
    response_headers.pop("Content-Encoding", None)
    response_headers.pop("Content-Length", None)

    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=response_headers,
    )

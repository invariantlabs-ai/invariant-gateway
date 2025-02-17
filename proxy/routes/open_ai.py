"""Proxy service to forward requests to the OpenAI APIs"""

import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from utils.constants import CLIENT_TIMEOUT, IGNORED_HEADERS
from utils.explorer import push_trace

ALLOWED_OPEN_AI_ENDPOINTS = {"chat/completions"}

proxy = APIRouter()

MISSING_INVARIANT_AUTH_API_KEY = "Missing invariant api key"
MISSING_AUTH_HEADER = "Missing authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported OpenAI endpoint"
FINISH_REASON_TO_PUSH_TRACE = ["stop", "length", "content_filter"]


def validate_headers(authorization: str = Header(None)):
    """Require the authorization header to be present"""
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
) -> Response:
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

    # The invariant-authorization header contains the Invariant API Key
    # "invariant-authorization": "Bearer <Invariant API Key>"
    # The authorization header contains the OpenAI API Key
    # "authorization": "Bearer <OpenAI API Key>"
    #
    # For some clients, it is not possible to pass a custom header
    # In such cases, the Invariant API Key is passed as part of the
    # authorization header with the OpenAI API key.
    # The header in that case becomes:
    # "authorization": "Bearer <OpenAI API Key>|invariant-auth: <Invariant API Key>"
    if request.headers.get(
        "invariant-authorization"
    ) is None and "|invariant-auth:" not in request.headers.get("authorization"):
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_API_KEY)

    if request.headers.get("invariant-authorization"):
        invariant_authorization = request.headers.get("invariant-authorization")
    else:
        authorization = request.headers.get("authorization")
        api_keys = authorization.split("|invariant-auth: ")
        invariant_authorization = f"Bearer {api_keys[1].strip()}"
        # Update the authorization header to pass the OpenAI API Key to the OpenAI API
        headers["authorization"] = f"{api_keys[0].strip()}"

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
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
            invariant_authorization,
        )
    async with client:
        response = await client.send(open_ai_request)
        return await handle_non_streaming_response(
            response, dataset_name, request_body_json, invariant_authorization
        )


async def stream_response(
    client: httpx.AsyncClient,
    open_ai_request: httpx.Request,
    dataset_name: str,
    request_body_json: dict[str, Any],
    invariant_authorization: str,
) -> StreamingResponse:
    """
    Handles streaming the OpenAI response to the client while building a merged_response
    The chunks are returned to the caller immediately
    The merged_response is built from the chunks as they are received
    It is sent to the Invariant Explorer at the end of the stream
    """

    async def event_generator() -> Any:
        # merged_response will be updated with the data from the chunks in the stream
        # At the end of the stream, this will be sent to the explorer
        merged_response = {
            "id": None,
            "object": "chat.completion",
            "created": None,
            "model": None,
            "choices": [],
            "usage": None,
        }
        # Each chunk in the stream contains a list called "choices" each entry in the list
        # has an index.
        # A choice has a field called "delta" which may contain a list called "tool_calls".
        # Maps the choice index in the stream to the index in the merged_response["choices"] list
        choice_mapping_by_index = {}
        # Combines the choice index and tool call index to uniquely identify a tool call
        tool_call_mapping_by_index = {}

        async with client.stream(
            "POST",
            open_ai_request.url,
            headers=open_ai_request.headers,
            content=open_ai_request.content,
        ) as response:
            if response.status_code != 200:
                yield json.dumps(
                    {"error": f"Failed to fetch response: {response.status_code}"}
                ).encode()
                return

            async for chunk in response.aiter_bytes():
                chunk_text = chunk.decode().strip()
                if not chunk_text:
                    continue

                # Yield chunk immediately to the client (proxy behavior)
                yield chunk

                # Process the chunk
                # This will update merged_response with the data from the chunk
                process_chunk_text(
                    chunk_text,
                    merged_response,
                    choice_mapping_by_index,
                    tool_call_mapping_by_index,
                )

            # Send full merged response to the explorer
            await push_to_explorer(
                dataset_name,
                merged_response,
                request_body_json,
                invariant_authorization,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def initialize_merged_response() -> dict[str, Any]:
    """Initializes the full response dictionary"""
    return {
        "id": None,
        "object": "chat.completion",
        "created": None,
        "model": None,
        "choices": [],
        "usage": None,
    }


def process_chunk_text(
    chunk_text: str,
    merged_response: dict[str, Any],
    choice_mapping_by_index: dict[int, int],
    tool_call_mapping_by_index: dict[str, dict[str, Any]],
) -> None:
    """Processes the chunk text and updates the merged_response to be sent to the explorer"""
    # Split the chunk text into individual JSON strings
    # A single chunk can contain multiple "data: " sections
    for json_string in chunk_text.split("\ndata: "):
        json_string = json_string.replace("data: ", "").strip()

        if not json_string or json_string == "[DONE]":
            continue

        try:
            json_chunk = json.loads(json_string)
        except json.JSONDecodeError:
            continue

        update_merged_response(
            json_chunk,
            merged_response,
            choice_mapping_by_index,
            tool_call_mapping_by_index,
        )


def update_merged_response(
    json_chunk: dict[str, Any],
    merged_response: dict[str, Any],
    choice_mapping_by_index: dict[int, int],
    tool_call_mapping_by_index: dict[str, dict[str, Any]],
) -> None:
    """Updates the merged_response with the data (content, tool_calls, etc.) from the JSON chunk"""
    merged_response["id"] = merged_response["id"] or json_chunk.get("id")
    merged_response["created"] = merged_response["created"] or json_chunk.get("created")
    merged_response["model"] = merged_response["model"] or json_chunk.get("model")

    for choice in json_chunk.get("choices", []):
        index = choice.get("index", 0)

        if index not in choice_mapping_by_index:
            choice_mapping_by_index[index] = len(merged_response["choices"])
            merged_response["choices"].append(
                {
                    "index": index,
                    "message": {"role": "assistant"},
                    "finish_reason": None,
                }
            )

        existing_choice = merged_response["choices"][choice_mapping_by_index[index]]
        delta = choice.get("delta", {})
        if choice.get("finish_reason"):
            existing_choice["finish_reason"] = choice["finish_reason"]

        update_existing_choice_with_delta(
            existing_choice, delta, tool_call_mapping_by_index, choice_index=index
        )


def update_existing_choice_with_delta(
    existing_choice: dict[str, Any],
    delta: dict[str, Any],
    tool_call_mapping_by_index: dict[str, dict[str, Any]],
    choice_index: int,
) -> None:
    """Updates the choice with the data from the delta"""
    content = delta.get("content")
    if content is not None:
        if "content" not in existing_choice["message"]:
            existing_choice["message"]["content"] = ""
        existing_choice["message"]["content"] += content

    if isinstance(delta.get("tool_calls"), list):
        if "tool_calls" not in existing_choice["message"]:
            existing_choice["message"]["tool_calls"] = []

        for tool in delta["tool_calls"]:
            tool_index = tool.get("index")
            tool_id = tool.get("id")
            name = tool.get("function", {}).get("name")
            arguments = tool.get("function", {}).get("arguments", "")

            if tool_index is None:
                continue

            choice_with_tool_call_index = f"{choice_index}-{tool_index}"

            if choice_with_tool_call_index not in tool_call_mapping_by_index:
                tool_call_mapping_by_index[choice_with_tool_call_index] = {
                    "index": tool_index,
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": "",
                    },
                }
                existing_choice["message"]["tool_calls"].append(
                    tool_call_mapping_by_index[choice_with_tool_call_index]
                )

            tool_call_entry = tool_call_mapping_by_index[choice_with_tool_call_index]

            if tool_id:
                tool_call_entry["id"] = tool_id

            if name:
                tool_call_entry["function"]["name"] = name

            if arguments:
                tool_call_entry["function"]["arguments"] += arguments

    finish_reason = delta.get("finish_reason")
    if finish_reason is not None:
        existing_choice["finish_reason"] = finish_reason


async def push_to_explorer(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    # Only push the trace to explorer if the message is an end turn message
    if (
        merged_response.get("choices")
        and merged_response["choices"][0].get("finish_reason")
        not in FINISH_REASON_TO_PUSH_TRACE
    ):
        return
    # Combine the messages from the request body and the choices from the OpenAI response
    messages = request_body.get("messages", [])
    messages += [choice["message"] for choice in merged_response.get("choices", [])]
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
    """Handles non-streaming OpenAI responses"""
    json_response = response.json()
    await push_to_explorer(
        dataset_name, json_response, request_body_json, invariant_authorization
    )

    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

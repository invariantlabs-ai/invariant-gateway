"""Gateway service to forward requests to the OpenAI APIs"""

import asyncio
import json
from typing import Any, Optional

import httpx
from common.config_manager import GatewayConfig, GatewayConfigManager
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from integrations.explorer import create_annotations_from_guardrails_errors, push_trace
from integrations.guardails import check_guardrails
from common.authorization import extract_authorization_from_headers
from common.request_context_data import RequestContextData

gateway = APIRouter()

MISSING_AUTH_HEADER = "Missing authorization header"
FINISH_REASON_TO_PUSH_TRACE = ["stop", "length", "content_filter"]
OPENAI_AUTHORIZATION_HEADER = "authorization"


def validate_headers(authorization: str = Header(None)):
    """Require the authorization header to be present"""
    if authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)


@gateway.post(
    "/{dataset_name}/openai/chat/completions",
    dependencies=[Depends(validate_headers)],
)
@gateway.post(
    "/openai/chat/completions",
    dependencies=[Depends(validate_headers)],
)
async def openai_chat_completions_gateway(
    request: Request,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),  # pylint: disable=unused-argument
) -> Response:
    """Proxy calls to the OpenAI APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, openai_api_key = extract_authorization_from_headers(
        request, dataset_name, OPENAI_AUTHORIZATION_HEADER
    )
    headers[OPENAI_AUTHORIZATION_HEADER] = openai_api_key

    request_body_bytes = await request.body()
    request_json = json.loads(request_body_bytes)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    open_ai_request = client.build_request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=request_body_bytes,
        headers=headers,
    )

    context = RequestContextData(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
        config=config,
    )

    if request_json.get("stream", False):
        return await stream_response(
            context,
            client,
            open_ai_request,
        )
    response = await client.send(open_ai_request)
    return await handle_non_streaming_response(
        context,
        response,
    )


async def stream_response(
    context: RequestContextData,
    client: httpx.AsyncClient,
    open_ai_request: httpx.Request,
) -> Response:
    """
    Handles streaming the OpenAI response to the client while building a merged_response
    The chunks are returned to the caller immediately
    The merged_response is built from the chunks as they are received
    It is sent to the Invariant Explorer at the end of the stream
    """

    response = await client.send(open_ai_request, stream=True)
    if response.status_code != 200:
        error_content = await response.aread()
        try:
            error_json = json.loads(error_content.decode("utf-8"))
            error_detail = error_json.get("error", "Unknown error from OpenAI API")
        except json.JSONDecodeError:
            error_detail = {"error": "Failed to parse OpenAI error response"}
        raise HTTPException(status_code=response.status_code, detail=error_detail)

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

        async for chunk in response.aiter_bytes():
            chunk_text = chunk.decode().strip()
            if not chunk_text:
                continue

            # Process the chunk
            # This will update merged_response with the data from the chunk
            process_chunk_text(
                chunk_text,
                merged_response,
                choice_mapping_by_index,
                tool_call_mapping_by_index,
            )

            # Check guardrails on the last chunk.
            if (
                chunk_text == "data: [DONE]"
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
                            "error": {
                                "message": "[Invariant] The response did not pass the guardrails",
                                "details": guardrails_execution_result,
                            }
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
                    yield f"data: {error_chunk}\n\n".encode()
                    return

            # Yield chunk to the client
            yield chunk

        # Send full merged response to the explorer
        # Don't block on the response from explorer
        if context.dataset_name:
            asyncio.create_task(push_to_explorer(context, merged_response))

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
    context: RequestContextData,
    merged_response: dict[str, Any],
    guardrails_execution_result: Optional[dict] = None,
) -> None:
    """Pushes the merged response to the Invariant Explorer"""
    # Only push the trace to explorer if the message is an end turn message
    # or if the guardrails check returned errors.
    guardrails_execution_result = guardrails_execution_result or {}
    guardrails_errors = guardrails_execution_result.get("errors", [])
    if guardrails_errors or not (
        merged_response.get("choices")
        and merged_response["choices"][0].get("finish_reason")
        not in FINISH_REASON_TO_PUSH_TRACE
    ):
        annotations = create_annotations_from_guardrails_errors(guardrails_errors)
        # Combine the messages from the request body and the choices from the OpenAI response
        messages = context.request_json.get("messages", [])
        messages += [choice["message"] for choice in merged_response.get("choices", [])]
        _ = await push_trace(
            dataset_name=context.dataset_name,
            invariant_authorization=context.invariant_authorization,
            messages=[messages],
            annotations=[annotations],
        )


async def get_guardrails_check_result(
    context: RequestContextData, json_response: dict[str, Any]
) -> dict[str, Any]:
    """Get the guardrails check result"""
    messages = list(context.request_json.get("messages", []))
    messages += [choice["message"] for choice in json_response.get("choices", [])]
    # TODO: Remove this once the guardrails API is fixed
    for message in messages:
        if "tool_calls" in message and message["tool_calls"] is None:
            message["tool_calls"] = []

    # Block on the guardrails check
    guardrails_execution_result = await check_guardrails(
        messages=messages,
        guardrails=context.config.guardrails,
        invariant_authorization=context.invariant_authorization,
    )
    return guardrails_execution_result


async def handle_non_streaming_response(
    context: RequestContextData, response: httpx.Response
) -> Response:
    """Handles non-streaming OpenAI responses"""
    try:
        json_response = response.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail="Invalid JSON response received from OpenAI API",
        ) from e
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=json_response.get("error", "Unknown error from OpenAI API"),
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

    return Response(
        content=response_string,
        status_code=response_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

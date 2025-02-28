"""Proxy service to forward requests to the Anthropic APIs"""

import json
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse
from utils.constants import CLIENT_TIMEOUT, IGNORED_HEADERS
from utils.explorer import push_trace
import routes.open_ai as open_ai
import litellm

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

HEADER_AUTHORIZATION = "x-api-key"


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
    dataset_name: str = None,
):
    """Proxy calls to the Anthropic APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"
    if request.headers.get(
        "invariant-authorization"
    ) is None and "|invariant-auth:" not in request.headers.get(HEADER_AUTHORIZATION):
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_API_KEY)

    if request.headers.get("invariant-authorization"):
        invariant_authorization = request.headers.get("invariant-authorization")
    else:
        authorization = request.headers.get(HEADER_AUTHORIZATION)
        api_keys = authorization.split("|invariant-auth: ")
        invariant_authorization = f"Bearer {api_keys[1].strip()}"
        # Update the authorization header to pass the OpenAI API Key to the OpenAI API
        headers[HEADER_AUTHORIZATION] = f"{api_keys[0].strip()}"

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
    else:
        response = await client.send(anthropic_request)
        return await handle_non_streaming_response(
            response, dataset_name, request_body_json, invariant_authorization
        )

async def push_to_explorer_with_openai_format(
    dataset_name: str,
    merged_response: dict[str, Any],
    request_body: dict[str, Any],
    invariant_authorization: str,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""

    # Combine the messages from the request body and the choices from the OpenAI response
    messages = request_body.get("messages", [])
    print("request body: ", request_body.get("messages", []))
    messages += [choice["message"] for choice in merged_response.get("choices", [])]

    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[messages],
        invariant_authorization=invariant_authorization,
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
    transformed_messages = connvert_anthropic_to_invariant_message_format(messages)
    _ = await push_trace(
        dataset_name=dataset_name,
        messages=[transformed_messages],
        invariant_authorization=invariant_authorization,
    )


async def handle_non_streaming_response(
    response: httpx.Response,
    dataset_name: str,
    request_body_json: dict[str, Any],
    invariant_authorization: str,
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
        request_messages_in_openai_format = connvert_anthropic_to_invariant_message_format(request_body_json.get("messages", []))
        response_in_openai_format = await convert_to_litellm_messages(response, request_body_json)
        await open_ai.push_to_explorer(
            dataset_name,
            response_in_openai_format.json(),
            {"messages": request_messages_in_openai_format},
            invariant_authorization,
        )
    return Response(
        content=json.dumps(json_response),
        status_code=response.status_code,
        media_type="application/json",
        headers=dict(response.headers),
    )

async def convert_to_litellm_messages(
    raw_response: httpx.Response,
    request_body_json: dict[str, Any],
    ) -> list[dict]:
    import uuid
    import time
    import os
    import tiktoken
    from litellm import ProviderConfigManager
    from litellm.utils import ModelResponse, LlmProviders
    from litellm.litellm_core_utils.litellm_logging import Logging

    model = request_body_json.get("model")
    model_response = ModelResponse()
    setattr(model_response, "usage", litellm.Usage())
    messages = request_body_json.get("messages")
    data = request_body_json
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    call_type = "completion" 
    stream = False
    custom_llm_provider="anthropic"

    litellm_call_id = str(uuid.uuid4())
    function_id = "None"
    optional_params = {}
    litellm_params = {}
    start_time = time.time()
    encoding = tiktoken.get_encoding("cl100k_base")
    json_mode = False
    
    logging_obj = Logging(
        function_id=function_id,
        model=model,
        messages=messages,
        call_type=call_type,
        stream=stream,
        start_time=start_time,
        litellm_call_id=litellm_call_id,
    )
    
    logging_obj.update_environment_variables(
        litellm_params=litellm_params,
        optional_params=optional_params,
    )

    config = ProviderConfigManager.get_provider_chat_config(
        model=model,
        provider=LlmProviders(custom_llm_provider),
    )

    response = config.transform_response(
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                json_mode=json_mode,
            )
    return response

async def handle_streaming_response(
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
    dataset_name: Optional[str],
    invariant_authorization: str,
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


def connvert_anthropic_to_invariant_message_format(
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

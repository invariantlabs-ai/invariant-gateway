"""Gateway service to forward requests to the OpenAI APIs"""

import json
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from gateway.common.authorization import extract_authorization_from_headers
from gateway.common.config_manager import (
    GatewayConfig,
    GatewayConfigManager,
    extract_guardrails_from_header,
)
from gateway.common.constants import (
    CLIENT_TIMEOUT,
    CONTENT_TYPE_EVENT_STREAM,
    CONTENT_TYPE_JSON,
    IGNORED_HEADERS,
)
from gateway.common.guardrails import GuardrailRuleSet
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import fetch_guardrails_from_explorer
from gateway.routes.instrumentation import (
    InstrumentedResponse,
    InstrumentedStreamingResponse,
)
from gateway.routes.base_provider import BaseProvider, ExtraItem, Replacement

gateway = APIRouter()

MISSING_AUTH_HEADER = "Missing authorization header"
FINISH_REASON_TO_PUSH_TRACE = ["stop", "length", "content_filter"]
OPENAI_AUTHORIZATION_HEADER = "authorization"


def validate_headers(authorization: str = Header(None)):
    """Require the authorization header to be present"""
    if authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)


def make_cors_response(request: Request, allow_methods: str) -> Response:
    """Returns a CORS response with the specified allowed methods"""
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": f"{allow_methods}, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400",
        },
    )


@gateway.options("/{dataset_name}/openai/chat/completions")
@gateway.options("/openai/chat/completions")
async def openai_chat_completions_options(request: Request):
    """Enables CORS for the OpenAI chat completions endpoint"""
    return make_cors_response(request, allow_methods="POST")


@gateway.options("/{dataset_name}/openai/models")
@gateway.options("/openai/models")
async def openai_models_options(request: Request):
    """Enables CORS for the OpenAI models endpoint"""
    return make_cors_response(request, allow_methods="GET")


@gateway.get("/{dataset_name}/openai/models")
@gateway.get("/openai/models")
async def openai_models_gateway(
    request: Request,
    dataset_name: str | None = None,
):
    """Proxy request to OpenAI /models endpoint"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    _, openai_api_key = extract_authorization_from_headers(
        request, dataset_name, OPENAI_AUTHORIZATION_HEADER
    )
    headers[OPENAI_AUTHORIZATION_HEADER] = "Bearer " + openai_api_key

    async with httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT)) as client:
        open_ai_request = client.build_request(
            "GET",
            "https://api.openai.com/v1/models",
            headers=headers,
        )
        result = await client.send(open_ai_request)
        return Response(
            content=result.content,
            status_code=result.status_code,
            headers=dict(result.headers),
        )


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
    dataset_name: str | None = None,
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),
    header_guardrails: GuardrailRuleSet = Depends(extract_guardrails_from_header),
) -> Response:
    """Proxy calls to the OpenAI chat completions endpoint"""

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, openai_api_key = extract_authorization_from_headers(
        request, dataset_name, OPENAI_AUTHORIZATION_HEADER
    )
    headers[OPENAI_AUTHORIZATION_HEADER] = "Bearer " + openai_api_key

    request_body_bytes = await request.body()
    request_json = json.loads(request_body_bytes)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    open_ai_request = client.build_request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=request_body_bytes,
        headers=headers,
    )

    # Fetch dataset guardrails
    dataset_guardrails = None
    if dataset_name:
        dataset_guardrails = await fetch_guardrails_from_explorer(
            dataset_name, invariant_authorization
        )

    # Create request context
    context = RequestContext.create(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
        guardrails=header_guardrails or dataset_guardrails,
        config=config,
        request=request,
    )

    provider = OpenAIProvider()

    # Handle streaming and non-streaming
    if request_json.get("stream", False):
        response = InstrumentedStreamingResponse(
            context=context,
            client=client,
            provider_request=open_ai_request,
            provider=provider,
        )
        return StreamingResponse(
            response.instrumented_event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
        )
    response = InstrumentedResponse(
        context=context,
        client=client,
        provider_request=open_ai_request,
        provider=provider,
    )
    return await response.instrumented_request()


class OpenAIProvider(BaseProvider):
    """Concrete implementation of BaseProvider for OpenAI"""

    def get_provider_name(self) -> str:
        return "openai"

    def combine_messages(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Combine request and response messages in OpenAI format"""
        messages = list(request_json.get("messages", []))
        if response_json:
            messages += [
                choice["message"] for choice in response_json.get("choices", [])
            ]
        return messages

    def create_metadata(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> dict[str, Any]:
        """OpenAI metadata creation"""
        metadata = {
            k: v for k, v in request_json.items() if k != "messages" and v is not None
        }
        metadata["via_gateway"] = True

        if response_json:
            metadata.update(
                {
                    key: value
                    for key, value in response_json.items()
                    if key in ("usage", "model") and value is not None
                }
            )
        return metadata

    def create_non_streaming_error_response(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
        status_code: int = 400,
    ) -> Replacement:
        """OpenAI non-streaming error format, replace the response with the error message"""
        return Replacement(
            Response(
                content=json.dumps(
                    {
                        "error": f"[Invariant] The {location} did not pass the guardrails",
                        "details": guardrails_execution_result,
                    }
                ),
                status_code=status_code,
                media_type=CONTENT_TYPE_JSON,
            ),
        )

    def create_error_chunk(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
    ) -> ExtraItem:
        """OpenAI streaming error format"""
        error_chunk = error_chunk = json.dumps(
            {
                "error": {
                    "message": f"[Invariant] The {location} did not pass the guardrails",
                    "details": guardrails_execution_result,
                }
            }
        )
        return ExtraItem(f"data: {error_chunk}\n\n".encode(), end_of_stream=True)

    def should_push_trace(
        self, merged_response: dict[str, Any], has_errors: bool
    ) -> bool:
        """OpenAI-specific push criteria"""

        return has_errors or not (
            merged_response.get("choices")
            and merged_response["choices"][0].get("finish_reason")
            not in FINISH_REASON_TO_PUSH_TRACE
        )

    def process_streaming_chunk(
        self, chunk: bytes, merged_response: dict[str, Any], chunk_state: dict[str, Any]
    ) -> None:
        """OpenAI streaming chunk processing"""
        chunk_text = chunk.decode().strip()
        if not chunk_text:
            return

        process_chunk_text(
            chunk_text,
            merged_response,
            chunk_state.get("choice_mapping_by_index", {}),
            chunk_state.get("tool_call_mapping_by_index", {}),
        )

    def is_streaming_complete(self, _: dict[str, Any], chunk_text: str = "") -> bool:
        """OpenAI completion detection"""
        return "data: [DONE]" in chunk_text

    def initialize_streaming_response(self) -> dict[str, Any]:
        """OpenAI streaming response structure"""
        return {
            "id": None,
            "object": "chat.completion",
            "created": None,
            "model": None,
            "choices": [],
            "usage": None,
        }

    def initialize_streaming_state(self) -> dict[str, Any]:
        """OpenAI streaming state"""
        return {"choice_mapping_by_index": {}, "tool_call_mapping_by_index": {}}


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

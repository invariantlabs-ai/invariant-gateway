"""Gateway service to forward requests to the Anthropic APIs"""

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
from gateway.converters.anthropic_to_invariant import (
    convert_anthropic_to_invariant_message_format,
)
from gateway.integrations.explorer import fetch_guardrails_from_explorer
from gateway.routes.base_provider import BaseProvider, ExtraItem, Replacement
from gateway.routes.instrumentation import (
    InstrumentedResponse,
    InstrumentedStreamingResponse,
)

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
    dataset_name: str | None = None,
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),
    header_guardrails: GuardrailRuleSet = Depends(extract_guardrails_from_header),
):
    """Proxy calls to the Anthropic APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, anthropic_api_key = extract_authorization_from_headers(
        request, dataset_name, ANTHROPIC_AUTHORIZATION_HEADER
    )
    headers[ANTHROPIC_AUTHORIZATION_HEADER] = anthropic_api_key

    request_body = await request.body()
    request_json = json.loads(request_body)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    anthropic_request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        data=request_body,
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

    provider = AnthropicProvider()

    # Handle streaming and non-streaming
    if request_json.get("stream"):
        response = InstrumentedStreamingResponse(
            context=context,
            client=client,
            provider_request=anthropic_request,
            provider=provider,
        )
        return StreamingResponse(
            response.instrumented_event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
        )
    response = InstrumentedResponse(
        context=context,
        client=client,
        provider_request=anthropic_request,
        provider=provider,
    )
    return await response.instrumented_request()


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
    We filter out the ping eventss

    """
    event_type = event.get("type")

    if event_type == MESSAGE_START:
        merged_response.update(**event.get("message"))
    elif event_type == CONTENT_BLOCK_START:
        index = event.get("index")
        if index >= len(merged_response.get("content")):
            merged_response["content"].append(event.get("content_block"))
        if event.get("content_block").get("type") == "tool_use":
            merged_response.get("content")[-1]["input"] = ""
    elif event_type == CONTENT_BLOCK_DELTA:
        index = event.get("index")
        delta = event.get("delta")
        if delta.get("type") == "text_delta":
            merged_response.get("content")[index]["text"] += delta.get("text")
        elif delta.get("type") == "input_json_delta":
            merged_response.get("content")[index]["input"] += delta.get("partial_json")
    elif event_type == MESSAGE_DELTA:
        merged_response["usage"].update(**event.get("usage"))


class AnthropicProvider(BaseProvider):
    """Concrete implementation of BaseProvider for Anthropic"""

    def get_provider_name(self) -> str:
        return "anthropic"

    def combine_messages(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Anthropic message combination with format conversion"""
        messages = []

        # Add system message if present (Anthropic-specific)
        if "system" in request_json:
            messages.append({"role": "system", "content": request_json.get("system")})

        messages.extend(request_json.get("messages", []))

        if response_json:
            messages.append(response_json)

        return convert_anthropic_to_invariant_message_format(messages)

    def create_metadata(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> dict[str, Any]:
        """Anthropic metadata creation"""
        metadata = {
            k: v
            for k, v in request_json.items()
            if k not in ["messages", "system"] and v is not None
        }
        metadata["via_gateway"] = True

        if response_json and response_json.get("usage"):
            metadata["usage"] = response_json["usage"]
        return metadata

    def create_non_streaming_error_response(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
        status_code: int = 400,
    ) -> Replacement:
        """Anthropic non-streaming error format"""
        error_chunk = json.dumps(
            {
                "error": {
                    "message": f"[Invariant] The {location} did not pass the guardrails",
                    "details": guardrails_execution_result,
                }
            }
        )
        return Replacement(
            Response(
                content=error_chunk,
                status_code=status_code,
                media_type=CONTENT_TYPE_JSON,
                headers={"content-type": CONTENT_TYPE_JSON},
            )
        )

    def create_error_chunk(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
    ) -> ExtraItem:
        """Anthropic streaming error format (SSE)"""
        error_chunk = json.dumps(
            {
                "error": {
                    "message": f"[Invariant] The {location} did not pass the guardrails",
                    "details": guardrails_execution_result,
                }
            }
        )
        return ExtraItem(
            f"event: error\ndata: {error_chunk}\n\n".encode(), end_of_stream=True
        )

    def should_push_trace(self, _1: dict[str, Any], _2: bool) -> bool:
        """Anthropic always pushes traces"""
        return True

    def process_streaming_chunk(
        self, chunk: bytes, merged_response: dict[str, Any], chunk_state: dict[str, Any]
    ) -> None:
        """
        Process the chunk and update the merged_response.
        Each chunk may contain multiple events, separated by double newlines.
        Each event has type and data fields, separated by a newline.
        It is possible that a chunk contains some incomplete events.

        Example:

        b'event: message_start\ndata: {"type":"message_start","message":
        {"id":"msg_01LkayzAaw7b7QkUAw91psyx","type":"message","role":"assistant"
        ,"model":"claude-3-5-sonnet-20241022","content":[],"stop_reason":null,
        "stop_sequence":null,"usage":{"input_tokens":20,"cache_creation_input_to'

        and

        b'kens":0,"cache_read_input_tokens":0,"output_tokens":1}}}\n\nevent: content_block_start
        \ndata: {"type":"content_block_start","index":0,"content_block"
        :{"type":"text","text":""} }\n\nevent: ping
        \ndata: {"type": "ping"}\n\nevent: content_block_delta
        \ndata: {"type":"content_block_delta","index":0,"delta":{"type":
        "text_delta","text":"Originally"} }\n\n'

        In this case the first chunk ends with 'cache_creation_input_to' which is
        continued in the next chunk.

        in this case we need to maintain a buffer of the incomplete events.
        We filter out the ping events and update a merged_response.
        """
        decoded_chunk = chunk.decode("utf-8", errors="replace")
        chunk_state["sse_buffer"] = chunk_state.get("sse_buffer", "") + decoded_chunk

        complete_events, incomplete_events = self._process_complete_events(
            chunk_state["sse_buffer"]
        )
        chunk_state["sse_buffer"] = incomplete_events

        # Update merged response with events
        for event in complete_events:
            try:
                lines = event.split("\n")
                event_type = None
                event_data = None

                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()

                if event_data and event_type != "ping":  # Ignore ping events
                    try:
                        event_json = json.loads(event_data)
                        update_merged_response(event_json, merged_response)
                    except json.JSONDecodeError:
                        pass
            except Exception:  # pylint: disable=broad-except
                pass

    def _process_complete_events(self, buffer: str) -> tuple[list[str], str]:
        """Streaming buffer processing"""
        if not buffer:
            return [], ""

        events = []
        remaining = buffer

        while "\n\n" in remaining:
            pos = remaining.find("\n\n")
            if pos >= 0:
                event = remaining[: pos + 2]
                remaining = remaining[pos + 2 :]
                if event.strip():
                    events.append(event)

        return events, remaining

    def is_streaming_complete(self, _: dict[str, Any], chunk_text: str = "") -> bool:
        """Anthropic streaming completion detection"""
        return "message_stop" in chunk_text

    def initialize_streaming_response(self) -> dict[str, Any]:
        """Anthropic starts with empty response"""
        return {}

    def initialize_streaming_state(self) -> dict[str, Any]:
        """Anthropic streaming state"""
        return {"sse_buffer": ""}

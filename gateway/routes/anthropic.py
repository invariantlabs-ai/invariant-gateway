"""Gateway service to forward requests to the Anthropic APIs"""

import asyncio
import json
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from starlette.responses import StreamingResponse

from gateway.common.authorization import extract_authorization_from_headers
from gateway.common.config_manager import (
    GatewayConfig,
    GatewayConfigManager,
    extract_guardrails_from_header,
)
from gateway.common.constants import (
    CLIENT_TIMEOUT,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_EVENT_STREAM,
    IGNORED_HEADERS,
)
from gateway.common.guardrails import GuardrailAction, GuardrailRuleSet
from gateway.common.request_context import RequestContext
from gateway.converters.anthropic_to_invariant import (
    convert_anthropic_to_invariant_message_format,
)
from gateway.integrations.explorer import (
    create_annotations_from_guardrails_errors,
    fetch_guardrails_from_explorer,
    push_trace,
)
from gateway.integrations.guardrails import (
    ExtraItem,
    InstrumentedResponse,
    InstrumentedStreamingResponse,
    Replacement,
    check_guardrails,
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
    dataset_name: str | None = None,  # This is None if the client doesn't want to push to Explorer
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),  # pylint: disable=unused-argument
    header_guardrails: GuardrailRuleSet = Depends(extract_guardrails_from_header),
):
    """Proxy calls to the Anthropic APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, anthopic_api_key = extract_authorization_from_headers(
        request, dataset_name, ANTHROPIC_AUTHORIZATION_HEADER
    )
    headers[ANTHROPIC_AUTHORIZATION_HEADER] = anthopic_api_key

    request_body = await request.body()
    request_json = json.loads(request_body)
    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    anthropic_request = client.build_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        data=request_body,
    )

    dataset_guardrails = None
    if dataset_name:
        # Get the guardrails for the dataset from explorer.
        dataset_guardrails = await fetch_guardrails_from_explorer(
            dataset_name, invariant_authorization
        )
    context = RequestContext.create(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
        guardrails=header_guardrails or dataset_guardrails,
        config=config,
        request=request,
    )
    if request_json.get("stream"):
        return await handle_streaming_response(context, client, anthropic_request)
    return await handle_non_streaming_response(context, client, anthropic_request)


def create_metadata(
    context: RequestContext, response_json: dict[str, Any]
) -> dict[str, Any]:
    """Creates metadata for the trace"""
    metadata = {k: v for k, v in context.request_json.items() if k != "messages"}
    metadata["via_gateway"] = True
    if response_json.get("usage"):
        metadata["usage"] = response_json.get("usage")
    return metadata


def combine_request_and_response_messages(
    context: RequestContext, response_json: dict[str, Any]
):
    """Combine the request and response messages"""
    messages = []
    if "system" in context.request_json:
        messages.append(
            {"role": "system", "content": context.request_json.get("system")}
        )
    messages.extend(context.request_json.get("messages", []))
    if len(response_json) > 0:
        messages.append(response_json)
    return messages


async def get_guardrails_check_result(
    context: RequestContext, action: GuardrailAction, response_json: dict[str, Any]
) -> dict[str, Any]:
    """Get the guardrails check result"""
    # Determine which guardrails to apply based on the action
    guardrails = (
        context.guardrails.logging_guardrails
        if action == GuardrailAction.LOG
        else context.guardrails.blocking_guardrails
    )
    if not guardrails:
        return {}

    messages = combine_request_and_response_messages(context, response_json)
    converted_messages = convert_anthropic_to_invariant_message_format(messages)

    # Block on the guardrails check
    guardrails_execution_result = await check_guardrails(
        messages=converted_messages,
        guardrails=guardrails,
        context=context,
    )
    return guardrails_execution_result


async def push_to_explorer(
    context: RequestContext,
    merged_response: dict[str, Any],
    guardrails_execution_result: dict | None = None,
) -> None:
    """Pushes the full trace to the Invariant Explorer"""
    guardrails_execution_result = guardrails_execution_result or {}
    annotations = create_annotations_from_guardrails_errors(
        guardrails_execution_result.get("errors", [])
    )

    # Execute the logging guardrails before pushing to Explorer
    logging_guardrails_execution_result = await get_guardrails_check_result(
        context,
        action=GuardrailAction.LOG,
        response_json=merged_response,
    )
    logging_annotations = create_annotations_from_guardrails_errors(
        logging_guardrails_execution_result.get("errors", [])
    )
    # Update the annotations with the logging guardrails
    annotations.extend(logging_annotations)

    # Combine the messages from the request body and Anthropic response
    messages = combine_request_and_response_messages(context, merged_response)
    converted_messages = convert_anthropic_to_invariant_message_format(messages)

    _ = await push_trace(
        dataset_name=context.dataset_name,
        messages=[converted_messages],
        invariant_authorization=context.invariant_authorization,
        metadata=[create_metadata(context, merged_response)],
        annotations=[annotations] if annotations else None,
    )


class InstrumentedAnthropicResponse(InstrumentedResponse):
    """Instrumented response for Anthropic API"""

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        anthropic_request: httpx.Request,
    ):
        super().__init__()
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.anthropic_request: httpx.Request = anthropic_request

        # response data
        self.response: httpx.Response | None = None
        self.response_string: str | None = None
        self.response_json: dict[str, Any] | None = None

        # guardrailing response (if any)
        self.guardrails_execution_result = {}

    async def on_start(self):
        """
        Check guardrails in a pipelined fashion, before processing the first
        chunk (for input guardrailing).
        """
        if self.context.guardrails:
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context, action=GuardrailAction.BLOCK, response_json={}
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    {
                        "error": {
                            "message": "[Invariant] The request did not pass the guardrails",
                            "details": self.guardrails_execution_result,
                        }
                    }
                )

                # Push annotated trace to the explorer - don't block on its response
                if self.context.dataset_name:
                    asyncio.create_task(
                        push_to_explorer(
                            self.context,
                            {},
                            self.guardrails_execution_result,
                        )
                    )

                # if we find something, we prevent the request from going through
                # and return an error instead
                return Replacement(
                    Response(
                        content=error_chunk,
                        status_code=400,
                        media_type=CONTENT_TYPE_JSON,
                        headers={"content-type": CONTENT_TYPE_JSON},
                    )
                )

    async def request(self):
        """Make the request to the Anthropic API."""
        self.response = await self.client.send(self.anthropic_request)

        try:
            response_json = self.response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=self.response.status_code,
                detail=(
                    "Invalid JSON response received from Anthropic: "
                    f"{self.response.text}, got error: {e}"
                ),
            ) from e
        if self.response.status_code != 200:
            raise HTTPException(
                status_code=self.response.status_code,
                detail=response_json.get("error", "Unknown error from Anthropic"),
            )

        self.response_json = response_json
        self.response_string = json.dumps(response_json)

        return self._make_response(
            content=self.response_string,
            status_code=self.response.status_code,
        )

    def _make_response(self, content: str, status_code: int):
        """Creates a new Response object with the correct headers and content"""
        assert self.response is not None, "response is None"

        updated_headers = self.response.headers.copy()
        updated_headers.pop("Content-Length", None)

        return Response(
            content=content,
            status_code=status_code,
            media_type=CONTENT_TYPE_JSON,
            headers=dict(updated_headers),
        )

    async def on_end(self):
        """
        Checks guardrails after the response is received, and asynchronously
        pushes to Explorer.
        """
        # ensure the response data is available
        assert self.response is not None, "response is None"
        assert self.response_json is not None, "response_json is None"
        assert self.response_string is not None, "response_string is None"

        if self.context.guardrails:
            # Block on the guardrails check
            guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.response_json,
            )
            if guardrails_execution_result.get("errors", []):
                guardrail_response_string = json.dumps(
                    {
                        "error": "[Invariant] The response did not pass the guardrails",
                        "details": guardrails_execution_result,
                    }
                )

                # push to explorer (if configured)
                if self.context.dataset_name:
                    # Push to Explorer - don't block on its response
                    asyncio.create_task(
                        push_to_explorer(
                            self.context,
                            self.response_json,
                            guardrails_execution_result,
                        )
                    )

                return Replacement(
                    self._make_response(
                        content=guardrail_response_string,
                        status_code=400,
                    )
                )

        # push to explorer (if configured)
        if self.context.dataset_name:
            # Push to Explorer - don't block on its response
            asyncio.create_task(
                push_to_explorer(
                    self.context, self.response_json, guardrails_execution_result
                )
            )


async def handle_non_streaming_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
) -> Response:
    """Handles non-streaming Anthropic responses"""
    response = InstrumentedAnthropicResponse(
        context=context,
        client=client,
        anthropic_request=anthropic_request,
    )

    return await response.instrumented_request()


class InstrumentedAnthropicStreamingResponse(InstrumentedStreamingResponse):
    """Instrumented streaming response for Anthropic API"""

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        anthropic_request: httpx.Request,
    ):
        super().__init__()

        # request parameters
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.anthropic_request: httpx.Request = anthropic_request

        # response data
        self.merged_response = {}

        # guardrailing response (if any)
        self.guardrails_execution_result = {}

        self.sse_buffer = ""  # Buffer for incomplete events

    async def on_start(self):
        """
        Check guardrails in a pipelined fashion, before processing the
        first chunk (for input guardrailing).
        """
        if self.context.guardrails:
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.merged_response,
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    {
                        "error": {
                            "message": "[Invariant] The request did not pass the guardrails",
                            "details": self.guardrails_execution_result,
                        }
                    }
                )

                # Push annotated trace to the explorer - don't block on its response
                if self.context.dataset_name:
                    asyncio.create_task(
                        push_to_explorer(
                            self.context,
                            self.merged_response,
                            self.guardrails_execution_result,
                        )
                    )

                # if we find something, we end the stream prematurely (end_of_stream=True)
                # and yield an error chunk instead of actually beginning the stream
                return ExtraItem(
                    f"event: error\ndata: {error_chunk}\n\n".encode(),
                    end_of_stream=True,
                )

    async def event_generator(self):
        """Actual streaming response generator"""
        response = await self.client.send(self.anthropic_request, stream=True)
        if response.status_code != 200:
            error_content = await response.aread()
            try:
                error_json = json.loads(error_content)
                error_detail = error_json.get("error", "Unknown error from Anthropic")
            except json.JSONDecodeError:
                error_detail = {
                    "error": "Failed to decode error response from Anthropic"
                }
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        # iterate over the response stream
        async for chunk in response.aiter_bytes():
            yield chunk

    async def on_chunk(self, chunk):
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
        # Decode the chunk and add to buffer
        decoded_chunk = chunk.decode("utf-8", errors="replace")
        self.sse_buffer += decoded_chunk

        # Process complete events from buffer
        complete_events, incomplete_events = self.process_complete_events(
            self.sse_buffer
        )
        self.sse_buffer = incomplete_events

        # Check if we've received message_stop in any events
        message_stop_received = False

        # Update the merged_response based on complete events
        for event in complete_events:
            try:
                if "event: message_stop" in event:
                    message_stop_received = True

                # Extract event data
                lines = event.split("\n")
                event_type = None
                event_data = None

                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()

                if event_data and event_type != "ping":  # Skip ping events
                    try:
                        event_json = json.loads(event_data)
                        update_merged_response(event_json, self.merged_response)
                    except json.JSONDecodeError as e:
                        print(
                            f"JSON parsing error in event: {e}. Event data: {event_data[:100]}...",
                            flush=True,
                        )
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error processing event: {e}", flush=True)

        # on last stream chunk, run output guardrails
        if message_stop_received and self.context.guardrails:
            # Block on the guardrails check
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.merged_response,
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "message": "[Invariant] The response did not pass the guardrails",
                            "details": self.guardrails_execution_result,
                        },
                    }
                )

                # yield an extra error chunk (without preventing the original chunk
                # to go through after,
                # so client gets the proper message_stop event still)
                return ExtraItem(
                    value=f"event: error\ndata: {error_chunk}\n\n".encode()
                )

    def process_complete_events(self, buffer):
        """Process the buffer and extract complete SSE events.

        Returns:
            tuple[list[str], str]: A tuple containing a list of
            complete events and the remaining buffer with incomplete events.
        """
        # Split on double newlines which separate SSE events
        if not buffer:
            return [], ""
        events = []
        remaining = buffer

        # Process events that are complete (ending with \n\n)
        while "\n\n" in remaining:
            pos = remaining.find("\n\n")
            if pos >= 0:
                event = remaining[: pos + 2]
                remaining = remaining[pos + 2 :]
                if event.strip():  # Skip empty events
                    events.append(event)

        return events, remaining

    async def on_end(self):
        """on_end: send full merged response to the explorer (if configured)"""
        # don't block on the response from explorer (.create_task)
        if self.context.dataset_name:
            asyncio.create_task(
                push_to_explorer(
                    self.context,
                    self.merged_response,
                    self.guardrails_execution_result,
                )
            )


async def handle_streaming_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    anthropic_request: httpx.Request,
) -> StreamingResponse:
    """Handles streaming Anthropic responses"""
    response = InstrumentedAnthropicStreamingResponse(
        context=context,
        client=client,
        anthropic_request=anthropic_request,
    )

    return StreamingResponse(
        response.instrumented_event_generator(), media_type=CONTENT_TYPE_EVENT_STREAM
    )


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

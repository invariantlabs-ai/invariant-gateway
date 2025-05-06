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
from gateway.common.constants import CLIENT_TIMEOUT, IGNORED_HEADERS
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
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
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
    guardrails_execution_result: Optional[dict] = None,
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
        self.response: Optional[httpx.Response] = None
        self.response_string: Optional[str] = None
        self.response_json: Optional[dict[str, Any]] = None

        # guardrailing response (if any)
        self.guardrails_execution_result = {}

    async def on_start(self):
        """Check guardrails in a pipelined fashion, before processing the first chunk (for input guardrailing)."""
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
                        media_type="application/json",
                        headers={"content-type": "application/json"},
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
                detail=f"Invalid JSON response received from Anthropic: {self.response.text}, got error{e}",
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
            media_type="application/json",
            headers=dict(updated_headers),
        )

    async def on_end(self):
        """Checks guardrails after the response is received, and asynchronously pushes to Explorer."""
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

    async def on_start(self):
        """Check guardrails in a pipelined fashion, before processing the first chunk (for input guardrailing)."""
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
        """Process the chunk and update the merged_response"""
        decoded_chunk = chunk.decode().strip()
        if not decoded_chunk:
            return

        # process chunk and extend the merged_response
        process_chunk(decoded_chunk, self.merged_response)

        # on last stream chunk, run output guardrails
        if "event: message_stop" in decoded_chunk and self.context.guardrails:
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

    async def on_end(self):
        """on_end: send full merged response to the exploree (if configured)"""
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
        response.instrumented_event_generator(), media_type="text/event-stream"
    )


def process_chunk(chunk: str, merged_response: dict[str, Any]) -> None:
    """
    Process the chunk of text and update the merged_response
    Example of chunk list can be find in:
    ../../resources/streaming_chunk_text/anthropic.txt
    """
    for text_block in chunk.split("\n\n"):
        # might be empty block
        if len(text_block.split("\ndata:")) > 1:
            event_text = text_block.split("\ndata:")[1]
            event = json.loads(event_text)
            update_merged_response(event, merged_response)


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

    """
    if event.get("type") == MESSAGE_START:
        merged_response.update(**event.get("message"))
    elif event.get("type") == CONTENT_BLOCK_START:
        index = event.get("index")
        if index >= len(merged_response.get("content")):
            merged_response["content"].append(event.get("content_block"))
        if event.get("content_block").get("type") == "tool_use":
            merged_response.get("content")[-1]["input"] = ""
    elif event.get("type") == CONTENT_BLOCK_DELTA:
        index = event.get("index")
        delta = event.get("delta")
        if delta.get("type") == "text_delta":
            merged_response.get("content")[index]["text"] += delta.get("text")
        elif delta.get("type") == "input_json_delta":
            merged_response.get("content")[index]["input"] += delta.get("partial_json")
    elif event.get("type") == MESSAGE_DELTA:
        merged_response["usage"].update(**event.get("usage"))

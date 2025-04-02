"""Gateway service to forward requests to the OpenAI APIs"""

import asyncio
import json
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from common.authorization import extract_authorization_from_headers
from common.config_manager import GatewayConfig, GatewayConfigManager
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from common.guardrails import GuardrailAction
from common.request_context import RequestContext
from integrations.explorer import (
    create_annotations_from_guardrails_errors,
    fetch_guardrails_from_explorer,
    push_trace,
)
from integrations.guardrails import (
    ExtraItem,
    InstrumentedResponse,
    InstrumentedStreamingResponse,
    check_guardrails,
    preload_guardrails,
)

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
async def openai_chat_completions_options(request: Request, dataset_name: str = None):
    """Enables CORS for the OpenAI chat completions endpoint"""
    return make_cors_response(request, allow_methods="POST")


@gateway.options("/{dataset_name}/openai/models")
@gateway.options("/openai/models")
async def openai_models_options(request: Request, dataset_name: str = None):
    """Enables CORS for the OpenAI models endpoint"""
    return make_cors_response(request, allow_methods="GET")


@gateway.get("/{dataset_name}/openai/models")
@gateway.get("/openai/models")
async def openai_models_gateway(
    request: Request,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
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

    dataset_guardrails = None
    if dataset_name:
        # Get the guardrails for the dataset
        dataset_guardrails = await fetch_guardrails_from_explorer(
            dataset_name, invariant_authorization
        )
    context = RequestContext.create(
        request_json=request_json,
        dataset_name=dataset_name,
        invariant_authorization=invariant_authorization,
        dataset_guardrails=dataset_guardrails,
        config=config,
    )
    asyncio.create_task(preload_guardrails(context))

    if request_json.get("stream", False):
        return await handle_stream_response(
            context,
            client,
            open_ai_request,
        )

    return await handle_non_stream_response(context, client, open_ai_request)


class InstrumentedOpenAIStreamResponse(InstrumentedStreamingResponse):
    """
    Does a streaming OpenAI completion request at the core, but also checks guardrails
    before (concurrent) and after the request.
    """

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        open_ai_request: httpx.Request,
    ):
        super().__init__()

        # request parameters
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.open_ai_request: httpx.Request = open_ai_request

        # guardrailing output (if any)
        self.guardrails_execution_result: Optional[dict] = None

        # merged_response will be updated with the data from the chunks in the stream
        # At the end of the stream, this will be sent to the explorer
        self.merged_response = {
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
        self.choice_mapping_by_index = {}
        # Combines the choice index and tool call index to uniquely identify a tool call
        self.tool_call_mapping_by_index = {}

    async def on_start(self):
        """
        Check guardrails in a pipelined fashion, before processing the first chunk
        (for input guardrailing).
        """
        if self.context.dataset_guardrails:
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
                    f"data: {error_chunk}\n\n".encode(),
                    end_of_stream=True,
                )

    async def on_chunk(self, chunk):
        """Processes each chunk of the stream and checks guardrails at the end of the stream"""
        # process and check each chunk
        chunk_text = chunk.decode().strip()
        if not chunk_text:
            return

        # Process the chunk
        # This will update merged_response with the data from the chunk
        process_chunk_text(
            chunk_text,
            self.merged_response,
            self.choice_mapping_by_index,
            self.tool_call_mapping_by_index,
        )

        # check guardrails at the end of the stream (on the '[DONE]' SSE chunk.)
        if "data: [DONE]" in chunk_text and self.context.dataset_guardrails:
            # Block on the guardrails check
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.merged_response,
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    {
                        "error": {
                            "message": "[Invariant] The response did not pass the guardrails",
                            "details": self.guardrails_execution_result,
                        }
                    }
                )

                # yield an extra error chunk (without preventing the original chunk to go through after)
                return ExtraItem(f"data: {error_chunk}\n\n".encode())

                # push will happen in on_end

    async def on_end(self):
        """Sends full merged response to the explorer."""
        # don't block on the response from explorer (.create_task)
        if self.context.dataset_name:
            asyncio.create_task(
                push_to_explorer(
                    self.context, self.merged_response, self.guardrails_execution_result
                )
            )

    async def event_generator(self):
        """Actual OpenAI stream response."""
        response = await self.client.send(self.open_ai_request, stream=True)
        if response.status_code != 200:
            error_content = await response.aread()
            try:
                error_json = json.loads(error_content.decode("utf-8"))
                error_detail = error_json.get("error", "Unknown error from OpenAI API")
            except json.JSONDecodeError:
                error_detail = {"error": "Failed to parse OpenAI error response"}
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        # stream out chunks
        async for chunk in response.aiter_bytes():
            yield chunk


async def handle_stream_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    open_ai_request: httpx.Request,
) -> Response:
    """
    Handles streaming the OpenAI response to the client while building a merged_response
    The chunks are returned to the caller immediately
    The merged_response is built from the chunks as they are received
    It is sent to the Invariant Explorer at the end of the stream
    """

    response = InstrumentedOpenAIStreamResponse(
        context,
        client,
        open_ai_request,
    )

    return StreamingResponse(
        response.instrumented_event_generator(), media_type="text/event-stream"
    )


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


def create_metadata(
    context: RequestContext, merged_response: dict[str, Any]
) -> dict[str, Any]:
    """Creates metadata for the trace"""
    metadata = {
        k: v
        for k, v in context.request_json.items()
        if k != "messages" and v is not None
    }
    metadata["via_gateway"] = True
    metadata.update(
        {
            key: value
            for key, value in merged_response.items()
            if key in ("usage", "model") and merged_response.get(key) is not None
        }
    )
    return metadata


async def push_to_explorer(
    context: RequestContext,
    merged_response: dict[str, Any],
    guardrails_execution_result: Optional[dict] = None,
) -> None:
    """Pushes the merged response to the Invariant Explorer"""
    # Only push the trace to explorer if the message is an end turn message
    # or if the guardrails check returned errors.
    guardrails_execution_result = guardrails_execution_result or {}
    guardrails_errors = guardrails_execution_result.get("errors", [])
    annotations = create_annotations_from_guardrails_errors(
        guardrails_errors, action="block"
    )
    # Execute the logging guardrails before pushing to Explorer
    logging_guardrails_execution_result = await get_guardrails_check_result(
        context,
        action=GuardrailAction.LOG,
        response_json=merged_response,
    )
    logging_annotations = create_annotations_from_guardrails_errors(
        logging_guardrails_execution_result.get("errors", []), action="log"
    )
    # Update the annotations with the logging guardrails
    annotations.extend(logging_annotations)

    if annotations or not (
        merged_response.get("choices")
        and merged_response["choices"][0].get("finish_reason")
        not in FINISH_REASON_TO_PUSH_TRACE
    ):
        # Combine the messages from the request body and the choices from the OpenAI response
        messages = list(context.request_json.get("messages", []))
        messages += [choice["message"] for choice in merged_response.get("choices", [])]
        _ = await push_trace(
            dataset_name=context.dataset_name,
            invariant_authorization=context.invariant_authorization,
            messages=[messages],
            annotations=[annotations],
            metadata=[create_metadata(context, merged_response)],
        )


async def get_guardrails_check_result(
    context: RequestContext,
    action: GuardrailAction,
    response_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the guardrails check result"""
    # Determine which guardrails to apply based on the action
    guardrails = (
        context.dataset_guardrails.logging_guardrails
        if action == GuardrailAction.LOG
        else context.dataset_guardrails.blocking_guardrails
    )
    if not guardrails:
        return {}

    messages = list(context.request_json.get("messages", []))
    if response_json is not None:
        messages += [choice["message"] for choice in response_json.get("choices", [])]

    # Block on the guardrails check
    guardrails_execution_result = await check_guardrails(
        messages=messages,
        guardrails=guardrails,
        invariant_authorization=context.invariant_authorization,
    )
    return guardrails_execution_result


class InstrumentedOpenAIResponse(InstrumentedResponse):
    """
    Does an OpenAI completion request at the core, but also checks guardrails
    before (concurrent) and after the request.
    """

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        open_ai_request: httpx.Request,
    ):
        super().__init__()

        # request parameters
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.open_ai_request: httpx.Request = open_ai_request

        # request outputs
        self.response: Optional[httpx.Response] = None
        self.response_json: Optional[dict[str, Any]] = None

        # guardrailing output (if any)
        self.guardrails_execution_result: Optional[dict] = None

    async def on_start(self):
        """
        Checks guardrails in a pipelined fashion, before processing
        the first chunk (for input guardrailing)
        """
        if self.context.dataset_guardrails:
            # block on the guardrails check
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context, action=GuardrailAction.BLOCK
            )
            if self.guardrails_execution_result.get("errors", []):
                # Push annotated trace to the explorer - don't block on its response
                if self.context.dataset_name:
                    asyncio.create_task(
                        push_to_explorer(
                            self.context,
                            {},
                            self.guardrails_execution_result,
                        )
                    )

                # replace the response with the error message
                return ExtraItem(
                    Response(
                        content=json.dumps(
                            {
                                "error": "[Invariant] The request did not pass the guardrails",
                                "details": self.guardrails_execution_result,
                            }
                        ),
                        status_code=400,
                        media_type="application/json",
                    ),
                    end_of_stream=True,
                )

    async def request(self):
        """Actual OpenAI request."""
        self.response = await self.client.send(self.open_ai_request)

        try:
            self.response_json = self.response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=self.response.status_code,
                detail="Invalid JSON response received from OpenAI API",
            ) from e
        if self.response.status_code != 200:
            raise HTTPException(
                status_code=self.response.status_code,
                detail=self.response_json.get("error", "Unknown error from OpenAI API"),
            )

        response_string = json.dumps(self.response_json)
        response_code = self.response.status_code

        return Response(
            content=response_string,
            status_code=response_code,
            media_type="application/json",
            headers=dict(self.response.headers),
        )

    async def on_end(self):
        """Postprocesses the OpenAI response and potentially replace it with a guardrails error."""

        # these two request outputs are guaranteed to be available by the time we reach
        # this point (after self.request() was executed)
        # nevertheless, we check for them to avoid any potential issues
        assert (
            self.response is not None
        ), "on_end called before 'self.response' was available"
        assert (
            self.response_json is not None
        ), "on_end called before 'self.response_json' was available"

        # extract original response status code
        response_code = self.response.status_code

        # if we have guardrails, check the response
        if self.context.dataset_guardrails:
            # run guardrails again, this time on request + response
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.response_json,
            )
            if self.guardrails_execution_result.get("errors", []):
                response_string = json.dumps(
                    {
                        "error": "[Invariant] The response did not pass the guardrails",
                        "details": self.guardrails_execution_result,
                    }
                )
                response_code = 400

                # Push annotated trace to the explorer - don't block on its response
                if self.context.dataset_name:
                    asyncio.create_task(
                        push_to_explorer(
                            self.context,
                            self.response_json,
                            self.guardrails_execution_result,
                        )
                    )

                # replace the response with the error message
                return ExtraItem(
                    Response(
                        content=response_string,
                        status_code=response_code,
                        media_type="application/json",
                    ),
                )

        # Push annotated trace to the explorer in any case - don't block on its response
        if self.context.dataset_name:
            asyncio.create_task(
                push_to_explorer(
                    self.context,
                    self.response_json,
                    # include any guardrailing errors if available
                    self.guardrails_execution_result,
                )
            )


async def handle_non_stream_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    open_ai_request: httpx.Request,
) -> Response:
    """Handles non-streaming OpenAI responses"""

    response = InstrumentedOpenAIResponse(
        context,
        client,
        open_ai_request,
    )

    return await response.instrumented_request()

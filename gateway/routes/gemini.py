"""Gateway service to forward requests to the Gemini APIs"""

import asyncio
import json
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from common.authorization import extract_authorization_from_headers
from common.config_manager import GatewayConfig, GatewayConfigManager
from common.constants import (
    CLIENT_TIMEOUT,
    IGNORED_HEADERS,
)
from common.guardrails import GuardrailAction
from common.request_context import RequestContext
from converters.gemini_to_invariant import convert_request, convert_response
from integrations.explorer import (
    create_annotations_from_guardrails_errors,
    fetch_guardrails_from_explorer,
    push_trace,
)
from integrations.guardrails import (
    ExtraItem,
    InstrumentedResponse,
    InstrumentedStreamingResponse,
    Replacement,
    preload_guardrails,
    check_guardrails,
)

gateway = APIRouter()

GEMINI_AUTHORIZATION_HEADER = "x-goog-api-key"


@gateway.post("/gemini/{api_version}/models/{model}:{endpoint}")
@gateway.post("/{dataset_name}/gemini/{api_version}/models/{model}:{endpoint}")
async def gemini_generate_content_gateway(
    request: Request,
    api_version: str,
    model: str,
    endpoint: str,
    dataset_name: str = None,  # This is None if the client doesn't want to push to Explorer
    alt: str = Query(
        None, title="Response Format", description="Set to 'sse' for streaming"
    ),
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),  # pylint: disable=unused-argument
) -> Response:
    """Proxy calls to the Gemini GenerateContent API"""
    if endpoint not in ["generateContent", "streamGenerateContent"]:
        return Response(
            content="Invalid endpoint - the only endpoints supported are: \
            /api/v1/gateway/gemini/<version>/models/<model-name>:generateContent or \
            /api/v1/gateway/<dataset-name>/gemini/<version>models/<model-name>:generateContent",
            status_code=400,
        )
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, gemini_api_key = extract_authorization_from_headers(
        request, dataset_name, GEMINI_AUTHORIZATION_HEADER
    )
    headers[GEMINI_AUTHORIZATION_HEADER] = gemini_api_key

    request_body_bytes = await request.body()
    request_json = json.loads(request_body_bytes)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    gemini_api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:{endpoint}"
    if alt == "sse":
        gemini_api_url += "?alt=sse"
    gemini_request = client.build_request(
        "POST",
        gemini_api_url,
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

    if alt == "sse" or endpoint == "streamGenerateContent":
        return await stream_response(
            context,
            client,
            gemini_request,
        )
    return await handle_non_streaming_response(
        context,
        client,
        gemini_request,
    )


class InstrumentedStreamingGeminiResponse(InstrumentedStreamingResponse):
    """Instrumented streaming response for Gemini API"""

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        gemini_request: httpx.Request,
    ):
        super().__init__()

        # request data
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.gemini_request: httpx.Request = gemini_request

        # Store the progressively merged response
        self.merged_response = {
            "candidates": [{"content": {"parts": []}, "finishReason": None}]
        }

        # guardrailing execution result (if any)
        self.guardrails_execution_result: Optional[dict[str, Any]] = None

    def make_refusal(
        self,
        location: Literal["request", "response"],
        guardrails_execution_result: dict[str, Any],
    ) -> dict:
        """Create a refusal response for the given request or response"""
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": f"[Invariant] The {location} did not pass the guardrails",
                            }
                        ],
                    }
                }
            ],
            "error": {
                "code": 400,
                "message": f"[Invariant] The {location} did not pass the guardrails",
                "details": guardrails_execution_result,
                "status": "INVARIANT_GUARDRAILS_VIOLATION",
            },
            "promptFeedback": {
                "blockReason": "SAFETY",
                "block_reason_message": f"[Invariant] The {location} did not pass the guardrails: "
                + json.dumps(guardrails_execution_result),
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_UNSPECIFIED",
                        "probability": "HIGH",
                        "blocked": True,
                    }
                ],
            },
        }

    async def on_start(self):
        """
        Check guardrails in a pipelined fashion, before processing the first chunk
        (for input guardrailing).
        """
        if self.context.dataset_guardrails:
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context, action=GuardrailAction.BLOCK, response_json={}
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    self.make_refusal("request", self.guardrails_execution_result)
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

                # if we find something, we end the stream prematurely (end_of_stream=True)
                # and yield an error chunk instead of actually beginning the stream
                return ExtraItem(
                    f"data: {error_chunk}\r\n\r\n".encode(), end_of_stream=True
                )

    async def event_generator(self):
        """Event generator for streaming responses"""
        response = await self.client.send(self.gemini_request, stream=True)

        if response.status_code != 200:
            error_content = await response.aread()
            try:
                error_json = json.loads(error_content.decode("utf-8"))
                error_detail = error_json.get("error", "Unknown error from Gemini API")
            except json.JSONDecodeError:
                error_detail = {"error": "Failed to parse Gemini error response"}
            raise HTTPException(status_code=response.status_code, detail=error_detail)

        async for chunk in response.aiter_bytes():
            yield chunk

    async def on_chunk(self, chunk):
        """Processes each chunk of the streaming response"""
        chunk_text = chunk.decode().strip()
        if not chunk_text:
            return

        # Parse and update merged_response incrementally
        process_chunk_text(self.merged_response, chunk_text)

        # runs on the last stream item
        if (
            self.merged_response.get("candidates", [])
            and self.merged_response.get("candidates")[0].get("finishReason", "")
            and self.context.dataset_guardrails
        ):
            # Block on the guardrails check
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.merged_response,
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    self.make_refusal("response", self.guardrails_execution_result)
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

                return ExtraItem(
                    value=f"data: {error_chunk}\r\n\r\n".encode(),
                    # for Gemini we have to end the stream prematurely, as the client SDK
                    # will not stop streaming when it encounters an error
                    end_of_stream=True,
                )

    async def on_end(self):
        """Runs when the stream ends."""

        # Push annotated trace to the explorer - don't block on its response
        if self.context.dataset_name:
            asyncio.create_task(
                push_to_explorer(
                    self.context,
                    self.merged_response,
                    self.guardrails_execution_result,
                )
            )


async def stream_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    gemini_request: httpx.Request,
) -> Response:
    """Handles streaming the Gemini response to the client"""

    response = InstrumentedStreamingGeminiResponse(
        context=context,
        client=client,
        gemini_request=gemini_request,
    )

    async def event_generator():
        async for chunk in response.instrumented_event_generator():
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def process_chunk_text(
    merged_response: dict[str, Any],
    chunk_text: str,
) -> None:
    """Processes the chunk text and updates the merged_response to be sent to the explorer"""
    # Split the chunk text into individual JSON strings
    # A single chunk can contain multiple "data: " sections
    for json_string in chunk_text.split("data: "):
        json_string = json_string.replace("data: ", "").strip()

        if not json_string:
            continue

        try:
            json_chunk = json.loads(json_string)
        except json.JSONDecodeError:
            print("Warning: Could not parse chunk:", json_string)

        update_merged_response(merged_response, json_chunk)


def update_merged_response(merged_response: dict[str, Any], chunk_json: dict) -> None:
    """Updates the merged response incrementally with a new chunk."""
    candidates = chunk_json.get("candidates", [])

    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if "text" in part:
                existing_parts = merged_response["candidates"][0]["content"]["parts"]
                if existing_parts and "text" in existing_parts[-1]:
                    existing_parts[-1]["text"] += part["text"]
                else:
                    existing_parts.append({"text": part["text"]})

            if "functionCall" in part:
                merged_response["candidates"][0]["content"]["parts"].append(
                    {"functionCall": part["functionCall"]}
                )

        if "role" in content:
            merged_response["candidates"][0]["content"]["role"] = content["role"]

        if "finishReason" in candidate:
            merged_response["candidates"][0]["finishReason"] = candidate["finishReason"]

    if "usageMetadata" in chunk_json:
        merged_response["usageMetadata"] = chunk_json["usageMetadata"]
    if "modelVersion" in chunk_json:
        merged_response["modelVersion"] = chunk_json["modelVersion"]


def create_metadata(
    context: RequestContext, response_json: dict[str, Any]
) -> dict[str, Any]:
    """Creates metadata for the trace"""
    metadata = {
        k: v
        for k, v in context.request_json.items()
        if k not in ("systemInstruction", "contents")
    }
    metadata["via_gateway"] = True
    metadata.update(
        {
            key: value
            for key, value in response_json.items()
            if key in ("usageMetadata", "modelVersion")
        }
    )
    return metadata


async def get_guardrails_check_result(
    context: RequestContext, action: GuardrailAction, response_json: dict[str, Any]
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

    converted_requests = convert_request(context.request_json)
    converted_responses = convert_response(response_json)

    # Block on the guardrails check
    guardrails_execution_result = await check_guardrails(
        messages=converted_requests + converted_responses,
        guardrails=guardrails,
        invariant_authorization=context.invariant_authorization,
    )
    return guardrails_execution_result


async def push_to_explorer(
    context: RequestContext,
    response_json: dict[str, Any],
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
        response_json=response_json,
    )
    logging_annotations = create_annotations_from_guardrails_errors(
        logging_guardrails_execution_result.get("errors", [])
    )
    # Update the annotations with the logging guardrails
    annotations.extend(logging_annotations)

    converted_requests = convert_request(context.request_json)
    converted_responses = convert_response(response_json)

    _ = await push_trace(
        dataset_name=context.dataset_name,
        messages=[converted_requests + converted_responses],
        invariant_authorization=context.invariant_authorization,
        metadata=[create_metadata(context, response_json)],
        annotations=[annotations] if annotations else None,
    )


class InstrumentedGeminiResponse(InstrumentedResponse):
    """Instrumented response for Gemini API"""

    def __init__(
        self,
        context: RequestContext,
        client: httpx.AsyncClient,
        gemini_request: httpx.Request,
    ):
        super().__init__()

        # request data
        self.context: RequestContext = context
        self.client: httpx.AsyncClient = client
        self.gemini_request: httpx.Request = gemini_request

        # response data
        self.response: Optional[httpx.Response] = None
        self.response_json: Optional[dict[str, Any]] = None

        # guardrails execution result (if any)
        self.guardrails_execution_result: Optional[dict[str, Any]] = None

    async def on_start(self):
        """
        Check guardrails in a pipelined fashion, before processing the first chunk
        (for input guardrailing).
        """
        if self.context.dataset_guardrails:
            self.guardrails_execution_result = await get_guardrails_check_result(
                self.context, action=GuardrailAction.BLOCK, response_json={}
            )
            if self.guardrails_execution_result.get("errors", []):
                error_chunk = json.dumps(
                    {
                        "error": {
                            "code": 400,
                            "message": "[Invariant] The request did not pass the guardrails",
                            "details": self.guardrails_execution_result,
                            "status": "INVARIANT_GUARDRAILS_VIOLATION",
                        },
                        "prompt_feedback": {
                            "blockReason": "SAFETY",
                            "safetyRatings": [
                                {
                                    "category": "HARM_CATEGORY_UNSPECIFIED",
                                    "probability": 0.0,
                                    "blocked": True,
                                }
                            ],
                        },
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

                # if we find something, we end the stream prematurely (end_of_stream=True)
                # and yield an error chunk instead of actually beginning the stream
                return Replacement(
                    Response(
                        content=error_chunk,
                        status_code=400,
                        media_type="application/json",
                        headers={
                            "Content-Type": "application/json",
                        },
                    )
                )

    async def request(self):
        """Makes the request to the Gemini API and return the response"""
        self.response = await self.client.send(self.gemini_request)

        response_string = self.response.text
        response_code = self.response.status_code

        try:
            self.response_json = self.response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=self.response.status_code,
                detail="Invalid JSON response received from Gemini API",
            ) from e
        if self.response.status_code != 200:
            raise HTTPException(
                status_code=self.response.status_code,
                detail=self.response_json.get("error", "Unknown error from Gemini API"),
            )

        return Response(
            content=response_string,
            status_code=response_code,
            media_type="application/json",
            headers=dict(self.response.headers),
        )

    async def on_end(self):
        response_string = json.dumps(self.response_json)
        response_code = self.response.status_code

        if self.context.dataset_guardrails:
            # Block on the guardrails check
            guardrails_execution_result = await get_guardrails_check_result(
                self.context,
                action=GuardrailAction.BLOCK,
                response_json=self.response_json,
            )
            if guardrails_execution_result.get("errors", []):
                response_string = json.dumps(
                    {
                        "error": {
                            "code": 400,
                            "message": "[Invariant] The response did not pass the guardrails",
                            "details": guardrails_execution_result,
                            "status": "INVARIANT_GUARDRAILS_VIOLATION",
                        },
                    }
                )
                response_code = 400

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
                    Response(
                        content=response_string,
                        status_code=response_code,
                        media_type="application/json",
                        headers=dict(self.response.headers),
                    )
                )

        # Otherwise, also push to Explorer - don't block on its response
        if self.context.dataset_name:
            asyncio.create_task(
                push_to_explorer(
                    self.context, self.response_json, guardrails_execution_result
                )
            )


async def handle_non_streaming_response(
    context: RequestContext,
    client: httpx.AsyncClient,
    gemini_request: httpx.Request,
) -> Response:
    """Handles non-streaming Gemini responses"""

    response = InstrumentedGeminiResponse(
        context=context,
        client=client,
        gemini_request=gemini_request,
    )

    return await response.instrumented_request()

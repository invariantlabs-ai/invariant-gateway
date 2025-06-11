"""Gateway service to forward requests to the Gemini APIs"""

import json
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response
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
from gateway.converters.gemini_to_invariant import (
    convert_request,
    convert_response,
)
from gateway.integrations.explorer import fetch_guardrails_from_explorer
from gateway.routes.base_provider import BaseProvider, ExtraItem, Replacement
from gateway.routes.instrumentation import (
    InstrumentedResponse,
    InstrumentedStreamingResponse,
)

gateway = APIRouter()

GEMINI_AUTHORIZATION_HEADER = "x-goog-api-key"
GEMINI_AUTHORIZATION_FALLBACK_HEADER = "authorization"


@gateway.post("/gemini/{api_version}/models/{model}:{endpoint}")
@gateway.post("/{dataset_name}/gemini/{api_version}/models/{model}:{endpoint}")
async def gemini_generate_content_gateway(
    request: Request,
    api_version: str,
    model: str,
    endpoint: str,
    dataset_name: str | None = None,
    alt: str = Query(
        None, title="Response Format", description="Set to 'sse' for streaming"
    ),
    config: GatewayConfig = Depends(GatewayConfigManager.get_config),
    header_guardrails: GuardrailRuleSet = Depends(extract_guardrails_from_header),
) -> Response:
    """Proxy calls to the Gemini APIs"""

    # Gemini endpoint validation
    if endpoint not in ["generateContent", "streamGenerateContent"]:
        return Response(
            content="Invalid endpoint - only generateContent and streamGenerateContent supported",
            status_code=400,
        )

    # Standard Gemini request setup
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in IGNORED_HEADERS + [GEMINI_AUTHORIZATION_FALLBACK_HEADER]
    }
    headers["accept-encoding"] = "identity"

    invariant_authorization, gemini_api_key = extract_authorization_from_headers(
        request,
        dataset_name,
        GEMINI_AUTHORIZATION_HEADER,
        [GEMINI_AUTHORIZATION_FALLBACK_HEADER],
    )
    headers[GEMINI_AUTHORIZATION_HEADER] = gemini_api_key

    request_body_bytes = await request.body()
    request_json = json.loads(request_body_bytes)

    client = httpx.AsyncClient(timeout=httpx.Timeout(CLIENT_TIMEOUT))
    gemini_api_url = (
        f"https://generativelanguage.googleapis.com/"
        f"{api_version}/models/"
        f"{model}:{endpoint}"
    )
    if alt == "sse":
        gemini_api_url += "?alt=sse"

    gemini_request = client.build_request(
        "POST",
        gemini_api_url,
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

    provider = GeminiProvider()

    # Handle streaming and non-streaming
    if alt == "sse" or endpoint == "streamGenerateContent":
        response = InstrumentedStreamingResponse(
            context=context,
            client=client,
            provider_request=gemini_request,
            provider=provider,
        )
        return StreamingResponse(
            response.instrumented_event_generator(),
            media_type=CONTENT_TYPE_EVENT_STREAM,
        )
    response = InstrumentedResponse(
        context=context,
        client=client,
        provider_request=gemini_request,
        provider=provider,
    )
    return await response.instrumented_request()


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


def make_refusal(
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


class GeminiProvider(BaseProvider):
    """Concrete implementation of BaseProvider for Gemini"""

    def get_provider_name(self) -> str:
        return "gemini"

    def combine_messages(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Gemini messages combination with format conversion"""
        converted_requests = convert_request(request_json)
        converted_responses = convert_response(response_json) if response_json else []

        return converted_requests + converted_responses

    def create_metadata(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> dict[str, Any]:
        """Gemini metadata creation"""
        metadata = {
            k: v
            for k, v in request_json.items()
            if k not in ["systemInstruction", "contents"] and v is not None
        }
        metadata["via_gateway"] = True

        if response_json:
            if response_json.get("usageMetadata"):
                metadata["usage"] = response_json["usageMetadata"]
            if response_json.get("modelVersion"):
                metadata["modelVersion"] = response_json["modelVersion"]
        return metadata

    def create_non_streaming_error_response(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
        status_code: int = 400,
    ) -> Replacement:
        """Gemini non-streaming error format"""
        error_chunk = json.dumps(
            {
                "error": {
                    "code": status_code,
                    "message": f"[Invariant] The {location} did not pass the guardrails",
                    "details": guardrails_execution_result,
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
        return Replacement(
            Response(
                content=error_chunk,
                status_code=400,
                media_type=CONTENT_TYPE_JSON,
                headers={
                    "Content-Type": CONTENT_TYPE_JSON,
                },
            )
        )

    def create_error_chunk(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
    ) -> ExtraItem:
        """Gemini streaming error format"""
        return ExtraItem(
            json.dumps(make_refusal(location, guardrails_execution_result)),
            end_of_stream=True,
        )

    def should_push_trace(
        self, merged_response: dict[str, Any], has_errors: bool
    ) -> bool:
        """Gemini push trace criteria"""
        return has_errors or (
            merged_response.get("candidates", [])
            and merged_response["candidates"][0].get("finishReason") is not None
        )

    def process_streaming_chunk(
        self, chunk: bytes, merged_response: dict[str, Any], _: dict[str, Any]
    ) -> None:
        """Gemini streaming chunk processing"""
        chunk_text = chunk.decode().strip()
        if not chunk_text:
            return

        for json_string in chunk_text.split("data: "):
            json_string = json_string.replace("data: ", "").strip()

            if not json_string:
                continue

            try:
                json_chunk = json.loads(json_string)
                update_merged_response(merged_response, json_chunk)
            except json.JSONDecodeError:
                continue

    def is_streaming_complete(
        self, merged_response: dict[str, Any], _: str = ""
    ) -> bool:
        """Gemini completion detection"""
        return (
            merged_response.get("candidates", [])
            and merged_response["candidates"][0].get("finishReason", "") != ""
        )

    def initialize_streaming_response(self) -> dict[str, Any]:
        """Gemini streaming response structure"""
        return {"candidates": [{"content": {"parts": []}, "finishReason": None}]}

    def initialize_streaming_state(self) -> dict[str, Any]:
        """Gemini has no additional state"""
        return {}

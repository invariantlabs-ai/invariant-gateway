"""Base LLM Provider Class for Invariant Gateway"""

import json
from typing import Any, Literal
from abc import ABC, abstractmethod

import httpx
from fastapi import HTTPException


class ExtraItem:
    """
    Return this class in a instrumented stream callback, to yield an extra item
    in the resulting stream.
    """

    def __init__(self, value, end_of_stream=False):
        self.value = value
        self.end_of_stream = end_of_stream

    def __str__(self):
        return f"<ExtraItem value={self.value} end_of_stream={self.end_of_stream}>"


class Replacement(ExtraItem):
    """
    Like ExtraItem, but used to replace the full request result in case of 'InstrumentedResponse'.
    """

    def __init__(self, value):
        super().__init__(value, end_of_stream=True)

    def __str__(self):
        return f"<Replacement value={self.value}>"


class BaseProvider(ABC):
    """
    Base Provider class that defines the protocol for all providers
    (e.g., OpenAI, Anthropic, Gemini).
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic', 'gemini')"""

    @abstractmethod
    def combine_messages(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Combine request and response messages in provider-specific way
        Handles message format conversion (e.g., Anthropic/Gemini converters)
        """

    @abstractmethod
    def create_metadata(
        self, request_json: dict[str, Any], response_json: dict[str, Any]
    ) -> dict[str, Any]:
        """Create provider-specific metadata"""

    @abstractmethod
    def create_non_streaming_error_response(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
        status_code: int = 400,
    ) -> Replacement:
        """Create provider-specific error response for non-streaming"""

    @abstractmethod
    def create_error_chunk(
        self,
        guardrails_execution_result: dict[str, Any],
        location: Literal["request", "response"] = "response",
    ) -> ExtraItem:
        """Create provider-specific error chunk for streaming"""

    @abstractmethod
    def should_push_trace(
        self, merged_response: dict[str, Any], has_errors: bool
    ) -> bool:
        """Provider-specific logic for when to push traces"""

    @abstractmethod
    def process_streaming_chunk(
        self, chunk: bytes, merged_response: dict[str, Any], chunk_state: dict[str, Any]
    ) -> None:
        """
        Process a streaming chunk and update merged_response
        chunk_state can hold provider-specific state (e.g., OpenAI's choice_mapping)
        """

    @abstractmethod
    def is_streaming_complete(
        self, merged_response: dict[str, Any], chunk_text: str = ""
    ) -> bool:
        """Determine if streaming is complete"""

    @abstractmethod
    def initialize_streaming_response(self) -> dict[str, Any]:
        """Initialize the merged response structure for streaming"""

    @abstractmethod
    def initialize_streaming_state(self) -> dict[str, Any]:
        """Initialize provider-specific state for streaming (e.g., OpenAI's mappings)"""

    def check_error_in_non_streaming_response(self, response: httpx.Response) -> None:
        """Check response status and parse JSON for non-streaming requests"""
        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Invalid JSON response received from {self.get_provider_name()}: "
                "{response.text}, error: {e}",
            ) from e
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response_json.get(
                    "error", f"Unknown error from {self.get_provider_name()}"
                ),
            )

    async def check_error_in_streaming_response(self, response: httpx.Response) -> None:
        """Check response status and parse JSON for streaming requests"""
        if response.status_code != 200:
            error_content = await response.aread()
            try:
                error_json = json.loads(error_content.decode("utf-8"))
                error_detail = error_json.get(
                    "error", f"Unknown error from {self.get_provider_name()}"
                )
            except json.JSONDecodeError:
                error_detail = {
                    "error": f"Failed to parse {self.get_provider_name()} error response"
                }

            raise HTTPException(status_code=response.status_code, detail=error_detail)

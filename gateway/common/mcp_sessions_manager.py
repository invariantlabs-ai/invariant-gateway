"""MCP Sessions Manager related classes"""

import asyncio
import contextlib
import os
import random

from typing import Any, Dict, List, Optional

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.append_messages import AppendMessagesRequest
from invariant_sdk.types.push_traces import PushTracesRequest
from pydantic import BaseModel, Field, PrivateAttr
from starlette.datastructures import Headers

from gateway.common.guardrails import GuardrailRuleSet, GuardrailAction
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import (
    create_annotations_from_guardrails_errors,
    fetch_guardrails_from_explorer,
)
from gateway.integrations.guardrails import check_guardrails

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


class McpSession(BaseModel):
    """
    Represents a single MCP session.
    """

    session_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id_to_method_mapping: Dict[int, str] = Field(default_factory=dict)
    explorer_dataset: str
    push_explorer: bool
    trace_id: Optional[str] = None
    last_trace_length: int = 0
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    guardrails: GuardrailRuleSet = Field(
        default_factory=lambda: GuardrailRuleSet(
            blocking_guardrails=[], logging_guardrails=[]
        )
    )
    # When tool calls are blocked, the error message is stored here
    # and sent to the client via the SSE stream.
    pending_error_messages: List[dict] = Field(default_factory=list)

    # Lock to maintain in-order pushes to explorer
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def load_guardrails(self) -> None:
        """
        Load guardrails for the session.

        This method fetches guardrails from the Invariant Explorer and assigns them to the session.
        """
        self.guardrails = await fetch_guardrails_from_explorer(
            self.explorer_dataset,
            "Bearer " + os.getenv("INVARIANT_API_KEY"),
        )

    def _deduplicate_annotations(self, new_annotations: list) -> list:
        """Deduplicate new_annotations using the annotations in the session."""
        deduped_annotations = []
        for annotation in new_annotations:
            if annotation not in self.annotations:
                deduped_annotations.append(annotation)
        return deduped_annotations

    @contextlib.asynccontextmanager
    async def session_lock(self):
        """
        Context manager for the session lock.

        Usage:
        async with session.session_lock():
            # Code that requires exclusive access to the session
        """
        async with self._lock:
            yield

    async def get_guardrails_check_result(
        self,
        message: dict,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> dict:
        """
        Check against guardrails of type action.
        """
        # Skip if no guardrails are configured for this action
        if not (
            (self.guardrails.blocking_guardrails and action == GuardrailAction.BLOCK)
            or (self.guardrails.logging_guardrails and action == GuardrailAction.LOG)
        ):
            return {}

        # Prepare context and select appropriate guardrails
        context = RequestContext.create(
            request_json={},
            dataset_name=self.explorer_dataset,
            invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
            guardrails=self.guardrails,
        )

        guardrails_to_check = (
            self.guardrails.blocking_guardrails
            if action == GuardrailAction.BLOCK
            else self.guardrails.logging_guardrails
        )

        result = await check_guardrails(
            messages=self.messages + [message],
            guardrails=guardrails_to_check,
            context=context,
        )
        return result

    async def add_message(
        self, message: Dict[str, Any], guardrails_result=Dict
    ) -> None:
        """
        Add a message to the session and optionally push to explorer.

        Args:
            message: The message to add
        """
        async with self.session_lock():
            annotations = []
            if guardrails_result and guardrails_result.get("errors", []):
                annotations = create_annotations_from_guardrails_errors(
                    guardrails_result.get("errors")
                )

            if self.guardrails.logging_guardrails:
                logging_guardrails_check_result = (
                    await self.get_guardrails_check_result(
                        message, action=GuardrailAction.LOG
                    )
                )
                if (
                    logging_guardrails_check_result
                    and logging_guardrails_check_result.get("errors", [])
                ):
                    annotations.extend(
                        create_annotations_from_guardrails_errors(
                            logging_guardrails_check_result["errors"]
                        )
                    )
            deduplicated_annotations = self._deduplicate_annotations(annotations)
            # pylint: disable=no-member
            self.messages.append(message)
            # If push_explorer is enabled, push the trace
            if self.push_explorer:
                await self._push_trace_update(deduplicated_annotations)

    async def _push_trace_update(self, deduplicated_annotations: list) -> None:
        """
        Push trace updates to the explorer.

        If a trace doesn't exist, create a new one. If it does, append new messages.

        This is an internal method that should only be called within a lock.
        """
        try:
            client = AsyncClient(
                api_url=os.getenv("INVARIANT_API_URL", DEFAULT_API_URL),
            )

            # If no trace exists, create a new one
            if not self.trace_id:
                # pylint: disable=no-member
                metadata = {"source": "mcp", "tools": self.metadata.get("tools", [])}
                if self.metadata.get("mcp_client_name"):
                    metadata["mcp_client"] = self.metadata.get("mcp_client_name")
                if self.metadata.get("mcp_server_name"):
                    metadata["mcp_server"] = self.metadata.get("mcp_server_name")

                response = await client.push_trace(
                    PushTracesRequest(
                        messages=[self.messages],
                        dataset=self.explorer_dataset,
                        metadata=[metadata],
                        annotations=[deduplicated_annotations],
                    )
                )
                self.trace_id = response.id[0]
            else:
                new_messages = self.messages[self.last_trace_length :]
                if new_messages:
                    await client.append_messages(
                        AppendMessagesRequest(
                            trace_id=self.trace_id,
                            messages=new_messages,
                            annotations=deduplicated_annotations,
                        )
                    )
            # pylint: disable=no-member
            self.annotations.extend(deduplicated_annotations)
            self.last_trace_length = len(self.messages)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[MCP SSE] Error pushing trace for session {self.session_id}: {e}")

    async def add_pending_error_message(self, error_message: dict) -> None:
        """
        Add a pending error message to the session.

        Args:
            error_message: The error message to add
        """
        async with self.session_lock():
            # pylint: disable=no-member
            self.pending_error_messages.append(error_message)

    async def get_pending_error_messages(self) -> List[dict]:
        """
        Get all pending error messages for the session.

        Returns:
            List[dict]: A list of pending error messages
        """
        async with self.session_lock():
            messages = list(self.pending_error_messages)
            self.pending_error_messages = []
            return messages


class SseHeaderAttributes(BaseModel):
    """
    A Pydantic model to represent header attributes.
    """

    push_explorer: bool
    explorer_dataset: str

    @classmethod
    def from_request_headers(cls, headers: Headers) -> "SseHeaderAttributes":
        """
        Create an instance from FastAPI request headers.

        Args:
            headers: FastAPI Request headers

        Returns:
            SseHeaderAttributes: An instance with values extracted from headers
        """
        # Extract and process header values
        project_name = headers.get("INVARIANT-PROJECT-NAME")
        push_explorer_header = headers.get("PUSH-INVARIANT-EXPLORER", "false").lower()

        # Determine explorer_dataset
        if project_name:
            explorer_dataset = project_name
        else:
            explorer_dataset = f"mcp-capture-{random.randint(1, 100)}"

        # Determine push_explorer
        push_explorer = push_explorer_header == "true"

        # Create and return instance
        return cls(push_explorer=push_explorer, explorer_dataset=explorer_dataset)


class McpSessionsManager:
    """
    A class to manage MCP sessions and their messages.
    """

    def __init__(self):
        self._sessions: dict[str, McpSession] = {}

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        return session_id in self._sessions

    async def initialize_session(
        self, session_id: str, sse_header_attributes: SseHeaderAttributes
    ) -> None:
        """Initialize a new session"""
        if session_id not in self._sessions:
            session = McpSession(
                session_id=session_id,
                explorer_dataset=sse_header_attributes.explorer_dataset,
                push_explorer=sse_header_attributes.push_explorer,
            )
            self._sessions[session_id] = session
            # Load guardrails for the session from the explorer
            await session.load_guardrails()

    def get_session(self, session_id: str) -> McpSession:
        """Get a session by ID"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} does not exist.")
        return self._sessions.get(session_id)

    async def add_message_to_session(
        self, session_id: str, message: Dict[str, Any], guardrails_result: dict
    ) -> None:
        """
        Add a message to a session and push to explorer if enabled.

        Args:
            session_id: The session ID
            message: The message to add
            guardrails_result: The result of the guardrails check
        """
        session = self.get_session(session_id)
        await session.add_message(message, guardrails_result)

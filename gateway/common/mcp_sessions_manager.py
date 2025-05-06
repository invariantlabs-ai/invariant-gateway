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

    # Lock to maintain in-order pushes to explorer
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

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

    async def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the session and optionally push to explorer.

        Args:
            message: The message to add
        """
        async with self.session_lock():
            # pylint: disable=no-member
            self.messages.append(message)
            # If push_explorer is enabled, push the trace
            if self.push_explorer:
                await self._push_trace_update()

    async def _push_trace_update(self) -> None:
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
                        )
                    )
            self.last_trace_length = len(self.messages)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[MCP SSE] Error pushing trace for session {self.session_id}: {e}")


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
        project_name = headers.get("PROJECT-NAME")
        push_explorer_header = headers.get("PUSH-EXPLORER", "false").lower()

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

    def initialize_session(
        self, session_id: str, sse_header_attributes: SseHeaderAttributes
    ) -> None:
        """Initialize a new session"""
        if session_id not in self._sessions:
            self._sessions[session_id] = McpSession(
                session_id=session_id,
                explorer_dataset=sse_header_attributes.explorer_dataset,
                push_explorer=sse_header_attributes.push_explorer,
            )

    def get_session(self, session_id: str) -> McpSession:
        """Get a session by ID"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} does not exist.")
        return self._sessions.get(session_id)

    async def add_message_to_session(
        self, session_id: str, message: Dict[str, Any]
    ) -> None:
        """
        Add a message to a session and push to explorer if enabled.

        Args:
            session_id: The session ID
            message: The message to add
        """
        session = self.get_session(session_id)
        await session.add_message(message)

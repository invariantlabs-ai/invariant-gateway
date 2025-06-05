"""MCP Sessions Manager related classes"""

import argparse
import asyncio
import contextlib
import getpass
import os
import random
import socket
from typing import Any

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.append_messages import AppendMessagesRequest
from invariant_sdk.types.push_traces import PushTracesRequest
from pydantic import BaseModel, Field, PrivateAttr
from starlette.datastructures import Headers

from gateway.common.constants import DEFAULT_API_URL
from gateway.common.guardrails import GuardrailAction, GuardrailRuleSet
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import (
    create_annotations_from_guardrails_errors,
    fetch_guardrails_from_explorer,
)
from gateway.integrations.guardrails import check_guardrails


def user_and_host() -> str:
    """Get the current user and hostname."""
    username = getpass.getuser()
    hostname = socket.gethostname()

    return f"{username}@{hostname}"

class McpAttributes(BaseModel):
    """
    A Pydantic model to represent MCP attributes.
    This can be initialized using HTTP headers for SSE and Streamable transports.
    This can also be initialized using CLI arguments for the Stdio transport.
    """

    push_explorer: bool
    explorer_dataset: str
    invariant_api_key: str | None = None
    verbose: bool | None = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_request_headers(cls, headers: Headers) -> "McpAttributes":
        """
        Create an instance from FastAPI request headers.

        Args:
            headers: FastAPI Request headers

        Returns:
            McpAttributes: An instance with values extracted from headers
        """
        # Extract and process header values
        project_name = headers.get("INVARIANT-PROJECT-NAME")
        push_explorer_header = headers.get("PUSH-INVARIANT-EXPLORER", "false").lower()
        invariant_api_key = headers.get("INVARIANT-API-KEY")

        # Determine explorer_dataset
        if project_name:
            explorer_dataset = project_name
        else:
            explorer_dataset = f"mcp-capture-{random.randint(1, 100)}"

        # Determine push_explorer
        push_explorer = push_explorer_header == "true"

        # Create and return instance
        return cls(
            push_explorer=push_explorer,
            explorer_dataset=explorer_dataset,
            invariant_api_key=invariant_api_key,
        )

    @classmethod
    def from_cli_args(cls, cli_args: list) -> "McpAttributes":
        """
        Create an instance from command line arguments.

        Args:
            cli_args: List of command line arguments

        Returns:
            McpAttributes: An instance with values extracted from CLI arguments
        """
        parser = argparse.ArgumentParser(description="MCP Gateway")
        parser.add_argument(
            "--project-name",
            help="Name of the Project from Invariant Explorer where we want to push the MCP traces. The guardrails are pulled from this project.",
            type=str,
            default=f"mcp-capture-{random.randint(1, 100)}",
        )
        parser.add_argument(
            "--push-explorer",
            help="Enable pushing traces to Invariant Explorer",
            action="store_true",
        )
        parser.add_argument(
            "--verbose",
            help="Enable verbose logging",
            action="store_true",
        )
        parser.add_argument(
            "--failure-response-format",
            help="The response format to use to communicate guardrail failures to the client (error: JSON-RPC error response; potentially invisible to the agent, content: JSON-RPC content response, visible to the agent)",
            type=str,
            default="error",
        )

        config, extra_args = parser.parse_known_args(cli_args)

        metadata: dict[str, Any] = {}
        for arg in extra_args:
            assert "=" in arg, f"Invalid extra metadata argument: {arg}"
            key, value = arg.split("=")
            assert key.startswith(
                "--metadata-"
            ), f"Invalid extra metadata argument: {arg}, must start with --metadata-"
            key = key[len("--metadata-") :]
            metadata[key] = value

        return cls(
            push_explorer=config.push_explorer,
            explorer_dataset=config.project_name,
            verbose=config.verbose,
            metadata=metadata,
        )


class McpSession(BaseModel):
    """
    Represents a single MCP session.
    """

    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    attributes: McpAttributes | None = None
    id_to_method_mapping: dict[int, str] = Field(default_factory=dict)
    trace_id: str | None = None
    last_trace_length: int = 0
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: GuardrailRuleSet = Field(
        default_factory=lambda: GuardrailRuleSet(
            blocking_guardrails=[], logging_guardrails=[]
        )
    )
    # When tool calls are blocked, the error message is stored here
    # and sent to the client via the SSE stream.
    pending_error_messages: list[dict] = Field(default_factory=list)

    # Lock to maintain in-order pushes to explorer
    # and other session-related operations
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def _get_invariant_api_key(self) -> str:
        """Get the Invariant API key for the session."""
        if self.attributes.invariant_api_key:
            return self.attributes.invariant_api_key
        return os.getenv("INVARIANT_API_KEY")

    def _get_invariant_authorization(self) -> str:
        """Get the Invariant authorization header for the session."""
        return "Bearer " + self._get_invariant_api_key()

    def _deduplicate_annotations(self, new_annotations: list) -> list:
        """Deduplicate new_annotations using the annotations in the session."""
        deduped_annotations = []
        for annotation in new_annotations:
            if annotation not in self.annotations:
                deduped_annotations.append(annotation)
        return deduped_annotations

    async def load_guardrails(self) -> None:
        """
        Load guardrails for the session.

        This method fetches guardrails from the Invariant Explorer and assigns them to the session.
        """
        self.guardrails = await fetch_guardrails_from_explorer(
            self.attributes.explorer_dataset,
            self._get_invariant_authorization(),
            # pylint: disable=no-member
            self.attributes.metadata.get("mcp_client"),
            self.attributes.metadata.get("mcp_server"),
        )

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

    def session_metadata(self) -> dict:
        """Generate metadata for the current session."""
        metadata = {
            "session_id": self.session_id,
            "system_user": user_and_host(),
            **(self.attributes.metadata or {}),
        }
        return metadata

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
            dataset_name=self.attributes.explorer_dataset,
            invariant_authorization=self._get_invariant_authorization(),
            guardrails=self.guardrails,
            guardrails_parameters={
                "metadata": self.session_metadata(),
                "action": action,
            },
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
        self, message: dict[str, Any], guardrails_result=dict
    ) -> None:
        """
        Add a message to the session and optionally push to explorer.

        Args:
            message: The message to add
            guardrails_result: The result of the guardrails check
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
            if self.attributes.push_explorer:
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
                api_key=self._get_invariant_api_key(),
            )

            # If no trace exists, create a new one
            if not self.trace_id:
                # default metadata
                metadata = {"source": "mcp"}
                # include MCP session metadata
                metadata.update(self.session_metadata())

                response = await client.push_trace(
                    PushTracesRequest(
                        messages=[self.messages],
                        dataset=self.attributes.explorer_dataset,
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
            print(f"[MCP] Error pushing trace for session {self.session_id}: {e}")

    async def add_pending_error_message(self, error_message: dict) -> None:
        """
        Add a pending error message to the session.

        Args:
            error_message: The error message to add
        """
        async with self.session_lock():
            # pylint: disable=no-member
            self.pending_error_messages.append(error_message)

    async def get_pending_error_messages(self) -> list[dict]:
        """
        Get all pending error messages for the session.

        Returns:
            list[dict]: A list of pending error messages
        """
        async with self.session_lock():
            messages = list(self.pending_error_messages)
            self.pending_error_messages = []
            return messages


class McpSessionsManager:
    """
    A class to manage MCP sessions and their messages.
    """

    def __init__(self):
        self._sessions: dict[str, McpSession] = {}
        # Dictionary to store per-session locks.
        # Used for session initialization and deletion.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Global lock to protect the locks dictionary itself
        self._global_lock = asyncio.Lock()

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        return session_id in self._sessions

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """
        Get a lock for a specific session ID, creating one if it doesn't exist.
        Uses the global lock to protect access to the locks dictionary.
        """
        async with self._global_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    async def cleanup_session_lock(self, session_id: str) -> None:
        """Remove a session lock when it's no longer needed"""
        async with self._global_lock:
            if session_id in self._session_locks:
                del self._session_locks[session_id]

    async def initialize_session(
        self, session_id: str, attributes: McpAttributes
    ) -> None:
        """Initialize a new session"""
        # Get the lock for this specific session
        session_lock = await self._get_session_lock(session_id)

        # Acquire the lock for this session
        async with session_lock:
            # Check again if session exists (it might have been created while waiting for the lock)
            if session_id not in self._sessions:
                session = McpSession(
                    session_id=session_id,
                    attributes=attributes,
                )
                self._sessions[session_id] = session
                # Load guardrails for the session from the explorer
                await session.load_guardrails()
            else:
                print(
                    f"Session {session_id} already exists, skipping initialization",
                    flush=True,
                )

    def get_session(self, session_id: str) -> McpSession:
        """Get a session by ID"""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} does not exist.")
        return self._sessions.get(session_id)

    async def add_message_to_session(
        self, session_id: str, message: dict[str, Any], guardrails_result: dict
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

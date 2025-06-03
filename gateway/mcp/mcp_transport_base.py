"""
MCP Transport Strategy Pattern Implementation

This module defines an abstract base class for MCP transports.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from gateway.mcp.constants import (
    MCP_METHOD,
    MCP_TOOL_CALL,
    MCP_LIST_TOOLS,
)
from gateway.mcp.mcp_sessions_manager import McpSessionsManager
from gateway.mcp.utils import (
    hook_tool_call,
    intercept_response,
    update_mcp_server_in_session_metadata,
    update_session_from_request,
)


class MCPTransportBase(ABC):
    """
    Abstract base class for MCP transport strategies.

    This class defines the common interface and shared functionality for all MCP transports,
    using the Template Method pattern for request/response processing.
    """

    def __init__(self, session_store: McpSessionsManager):
        self.session_store = session_store

    async def process_outgoing_request(
        self, session_id: str, request_data: dict[str, Any]
    ) -> Tuple[dict[str, Any], bool]:
        """
        Template method for processing outgoing requests to MCP server.

        Returns:
            Tuple[processed_request_data, is_blocked]
        """
        # Update session with request information
        session = self.session_store.get_session(session_id)
        update_session_from_request(session, request_data)

        # Refresh guardrails
        await session.load_guardrails()

        # Check if request should be intercepted for guardrails
        if self._should_intercept_request(request_data):
            return await self._intercept_outgoing_request(session_id, request_data)

        return request_data, False

    async def process_incoming_response(
        self, session_id: str, response_data: dict[str, Any]
    ) -> Tuple[dict[str, Any], bool]:
        """
        Template method for processing incoming responses from MCP server.

        Returns:
            Tuple[processed_response, is_blocked]
        """
        # Update session with server information
        session = self.session_store.get_session(session_id)
        update_mcp_server_in_session_metadata(session, response_data)

        # Intercept and apply guardrails to response
        return await intercept_response(session_id, self.session_store, response_data)

    def _should_intercept_request(self, request_data: dict[str, Any]) -> bool:
        """Check if request should be intercepted for guardrails."""
        method = request_data.get(MCP_METHOD)
        return method in [MCP_TOOL_CALL, MCP_LIST_TOOLS]

    async def _intercept_outgoing_request(
        self, session_id: str, request_data: dict[str, Any]
    ) -> Tuple[dict[str, Any], bool]:
        """Common request interception logic for guardrails."""
        method = request_data.get(MCP_METHOD)

        interception_result = request_data
        is_blocked = False
        if method == MCP_TOOL_CALL:
            interception_result, is_blocked = await hook_tool_call(
                session_id, self.session_store, request_data
            )
        elif method == MCP_LIST_TOOLS:
            interception_result, is_blocked = await hook_tool_call(
                session_id=session_id,
                session_store=self.session_store,
                request_body={
                    "id": request_data.get("id"),
                    "method": MCP_LIST_TOOLS,
                    "params": {"name": MCP_LIST_TOOLS, "arguments": {}},
                },
            )

        return interception_result, is_blocked

    @abstractmethod
    async def initialize_session(self, *args, **kwargs) -> str:
        """Initialize a session for this transport type."""

    @abstractmethod
    async def handle_communication(self, *args, **kwargs) -> Any:
        """Handle the main communication for this transport."""

"""Context manager for MCP (Model Context Protocol) gateway."""

import argparse
import os
import random
import uuid
from typing import Dict

from gateway.integrations.explorer import (
    fetch_guardrails_from_explorer,
)
from gateway.common.guardrails import GuardrailRuleSet


class McpContext:
    """Singleton class to manage MCP context and state."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(McpContext, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cli_args: list):
        if not hasattr(self, "_initialized"):
            self._initialized = False
        if self._initialized:
            return

        config, extra_args = self._parse_cli_args(cli_args)

        # The project name is used to identify the dataset in Invariant Explorer.
        self.explorer_dataset = config.project_name
        # whether to push traces to Invariant Explorer
        self.push_explorer = config.push_explorer
        # the format to use to communicate guardrail failures to the client
        self.failure_response_format = config.failure_response_format
        # verbose logging of in/out
        self.verbose = config.verbose

        # trace of this MCP session
        self.trace = []
        # tools available to the MCP server
        self.tools = []

        # configured guardrails
        self.guardrails = GuardrailRuleSet(
            blocking_guardrails=[], logging_guardrails=[]
        )

        # parsed from CLI (all --metadata-* args)
        self.extra_metadata: Dict[str, str] = {}
        for arg in extra_args:
            assert "=" in arg, f"Invalid extra metadata argument: {arg}"
            key, value = arg.split("=")
            assert key.startswith(
                "--metadata-"
            ), f"Invalid extra metadata argument: {arg}, must start with --metadata-"
            key = key[len("--metadata-") :]
            self.extra_metadata[key] = value

        # captured from MCP calls/responses
        self.mcp_client_name = ""
        self.mcp_server_name = ""

        # We send the same trace messages for guardrails analysis multiple times.
        # We need to deduplicate them before sending to the explorer.
        self.annotations = []
        self.trace_id = None
        self.local_session_id = str(uuid.uuid4())
        self.last_trace_length = 0
        self.id_to_method_mapping = {}
        self._initialized = True

    def _parse_cli_args(self, cli_args: list) -> argparse.Namespace:
        """Parse command line arguments."""
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
            help="The response format to use to communicate guardrail failures to the client (error: JSON-RPC error response; potentially invisble to the agent, content: JSON-RPC content response, visible to the agent)",
            type=str,
            default="content",
        )

        return parser.parse_known_args(cli_args)

    async def load_guardrails(self):
        """Run async setup logic (e.g. fetching guardrails)."""
        self.guardrails = await fetch_guardrails_from_explorer(
            self.explorer_dataset, "Bearer " + os.getenv("INVARIANT_API_KEY"),
            self.mcp_client_name,
            self.mcp_server_name,
        )

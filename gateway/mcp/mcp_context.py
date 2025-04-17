"""Context manager for MCP (Model Context Protocol) gateway."""

import argparse
import os
import random

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

        config = self._parse_cli_args(cli_args)
        # The project name is used to identify the dataset in Invariant Explorer.
        self.explorer_dataset = config.project_name
        self.push_explorer = config.push_explorer
        self.trace = []
        self.tools = []
        self.guardrails = GuardrailRuleSet(
            blocking_guardrails=[], logging_guardrails=[]
        )
        # We send the same trace messages for guardrails analysis multiple times.
        # We need to deduplicate them before sending to the explorer.
        self.annotations = []
        self.trace_id = None
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

        return parser.parse_args(cli_args)

    async def load_guardrails(self):
        """Run async setup logic (e.g. fetching guardrails)."""
        self.guardrails = await fetch_guardrails_from_explorer(
            self.explorer_dataset, "Bearer " + os.getenv("INVARIANT_API_KEY")
        )

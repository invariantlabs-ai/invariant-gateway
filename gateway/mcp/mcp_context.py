"""Context manager for MCP (Model Context Protocol) gateway."""

import atexit
import os
import sys
from invariant_sdk.client import Client


class McpContext:
    """Singleton class to manage MCP context and state."""

    _instance = None

    def __new__(cls):
        """Control instance creation to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(McpContext, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the singleton instance with default values (only once)."""
        # Define _initialized attribute explicitly at the beginning to avoid warnings
        # This is redundant but prevents warnings about accessing before definition
        if not hasattr(self, "_initialized"):
            self._initialized = False

        if self._initialized:
            return

        def setup_logging(self):
            """Set up logging to a file in the user's home directory.

            Uses proper resource management to ensure the file is closed on program exit.
            """
            os.makedirs(
                os.path.join(os.path.expanduser("~"), ".invariant"), exist_ok=True
            )
            log_path = os.path.join(os.path.expanduser("~"), ".invariant", "mcp.log")
            self.log_out = open(log_path, "a", buffering=1, encoding="utf-8")
            atexit.register(self.log_out.close)
            sys.stderr = self.log_out

        self.client = Client()
        self.explorer_dataset = "mcp-capture"
        self.trace = []
        self.tools = []
        self.trace_id = None
        self.last_trace_length = 0
        self.guardrails = None
        self.id_to_method_mapping = {}
        setup_logging(self)
        # Mark as initialized
        self._initialized = True

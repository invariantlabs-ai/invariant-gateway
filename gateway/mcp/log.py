"""Cusym log configuration."""

import os
import sys

from builtins import print as builtins_print

os.makedirs(os.path.join(os.path.expanduser("~"), ".invariant"), exist_ok=True)
MCP_LOG_FILE = open(
    os.path.join(os.path.expanduser("~"), ".invariant", "mcp.log"),
    "a",
    buffering=1,
)
sys.stderr = MCP_LOG_FILE


def mcp_log(*args, **kwargs) -> None:
    """Custom print function to redirect output to log_out."""
    builtins_print(*args, **kwargs, file=MCP_LOG_FILE, flush=True)

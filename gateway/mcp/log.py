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

def format_errors_in_response(errors: list[dict]) -> str:
    """Format a list of errors in a response string."""

    def format_error(error: dict) -> str:
        msg = " ".join(error.get("args", []))
        msg += " ".join([f"{k}={v}" for k, v in error.get("kwargs", {}).items()])
        msg += f" ([{error.get('guardrail', {}).get('id', 'unknown-guardrail')}] {error.get('guardrail', {}).get('name', 'unknown guardrail')})"
        return msg

    return ", ".join([format_error(error) for error in errors])
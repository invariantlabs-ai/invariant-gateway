"""Script is used to run actions using the Invariant Gateway."""

import sys

from gateway.mcp import mcp


def main():
    """Entry point for the Invariant Gateway."""
    actions = {
        "mcp": "Runs the Invariant Gateway against MCP (Model Context Protocol) servers with guardrailing and push to Explorer features",
        "llm": "Runs the Invariant Gateway against LLM providers with guardrailing and push to Explorer features",
        "help": "Shows this help message",
    }

    def _help():
        """_prints the help message."""
        for verb, description in actions.items():
            print(f"{verb}: {description}")

    if len(sys.argv) < 2:
        _help()
        sys.exit(1)

    verb = sys.argv[1]
    if verb == "mcp":
        return mcp.execute(sys.argv[2:])
    if verb == "llm":
        return 1
    if verb == "help":
        _help()
        return 0
    print(f"[gateway/__main__.py] Unknown action: {verb}")
    return 1

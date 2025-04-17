"""Script is used to run actions using the Invariant Gateway."""

import asyncio
import signal
import sys

from gateway.mcp import mcp


# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    sys.exit(0)


def print_help():
    """Prints the help message."""
    actions = {
        "mcp": "Runs the Invariant Gateway against MCP (Model Context Protocol) servers with guardrailing and push to Explorer features.",
        "llm": "Runs the Invariant Gateway against LLM providers with guardrailing and push to Explorer features. Not implemented yet.",
        "help": "Shows this help message.",
    }

    for verb, description in actions.items():
        print(f"{verb}: {description}")


def main():
    """Entry point for the Invariant Gateway."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    verb = sys.argv[1]
    if verb == "mcp":
        return asyncio.run(mcp.execute(sys.argv[2:]))
    if verb == "llm":
        print("[gateway/__main__.py] 'llm' action is not implemented yet.")
        return 1
    if verb == "help":
        print_help()
        return 0
    print(f"[gateway/__main__.py] Unknown action: {verb}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

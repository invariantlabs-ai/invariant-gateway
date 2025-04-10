"""Script is used to run actions using the Invariant Gateway."""

import sys


def main():
    """Entry point for the Invariant Gateway."""
    actions = {
        "mcp": "Runs the Invariant Gateway against MCP (Model Context Protocol) servers with guardrailing and push to Explorer features",
        "llm": "Runs the Invariant Gateway against LLM providers with guardrailing and push to Explorer features",
        "help": "Shows this help message",
    }

    def _help():
        """Prints the help message."""
        print("\nSupported Commands by invariant-gateway:\n")
        for verb, description in actions.items():
            print(f"{verb}: {description}")

    if len(sys.argv) < 2:
        _help()
        sys.exit(1)

    verb = sys.argv[1]
    if verb == "mcp":
        # Use sys.argv[2:] to pass arguments to the MCP gateway
        return 0
    if verb == "llm":
        # Use sys.argv[2:] to pass arguments to the LLM gateway
        print("LLM gateway via the invariant-gateway command is not implemented yet.")
        return 1
    if verb == "help":
        _help()
        return 0
    print(f"Unknown action: {verb}")
    return 1

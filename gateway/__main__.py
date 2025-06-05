"""Script is used to run actions using the Invariant Gateway."""

import asyncio
import os
import signal
import subprocess
import sys
import time

from gateway.mcp import stdio as mcp_stdio
from gateway.mcp.log import mcp_log


LOCAL_COMPOSE_FILE = "gateway/docker-compose.local.yml"


# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    sys.exit(0)


def print_help():
    """Prints the help message."""
    actions = {
        "mcp": """
        Runs the Invariant Gateway against MCP (Model Context Protocol) stdio servers with guardrailing and push to Explorer features.
        """,
        "server": """
        Runs the Invariant Gateway server locally providing guardrailing and push to Explorer features.
        Should be called with one of the following subcommands: build, up, down, logs.
        A guardrails file can be passed with the flag: --guardrails-file=/path/to/guardrails/file.
        """,
        "help": "Shows this help message.",
    }

    for verb, description in actions.items():
        print(f"{verb}: {description}")


def ensure_network_exists(network_name: str = "invariant-explorer-web") -> bool:
    """Ensure the Docker network exists."""
    try:
        # Check if network exists
        result = subprocess.run(
            ["docker", "network", "inspect", network_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode != 0:
            print(f"Creating Docker network: {network_name}")
            subprocess.run(
                ["docker", "network", "create", network_name],
                check=True,
            )

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating Docker network: {e}")
        return False


def setup_guardrails(guardrails_file_path: str | None = None) -> bool:
    """Configure guardrails if specified."""
    if not guardrails_file_path:
        return True

    if not os.path.isfile(guardrails_file_path):
        print(
            f"Error: Specified guardrails file does not exist: {guardrails_file_path}"
        )
        return False

    # Convert to absolute path
    guardrails_file_path = os.path.realpath(guardrails_file_path)
    os.environ["GUARDRAILS_FILE_PATH"] = guardrails_file_path

    # Check if INVARIANT_API_KEY is set
    if not os.environ.get("INVARIANT_API_KEY"):
        print(
            "Error: A guardrails file is specified, but INVARIANT_API_KEY env var is not set. "
            "This is required to validate guardrails."
        )
        return False

    return True


def build():
    """Build Docker containers using docker-compose."""
    try:
        print(f"Building using docker-compose file: {LOCAL_COMPOSE_FILE}")
        subprocess.run(
            ["docker", "compose", "-f", str(LOCAL_COMPOSE_FILE), "build"],
            check=True,
        )
        print("Build completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building containers: {e}")
        return False


def up(guardrails_file_path: str | None = None):
    """Set up the local server for the Invariant Gateway."""
    # Ensure network exists
    if not ensure_network_exists():
        return 1

    # Set up guardrails
    if not setup_guardrails(guardrails_file_path=guardrails_file_path):
        return 1

    # Run the server
    try:
        # Start containers
        print(f"Starting containers using docker-compose file: {LOCAL_COMPOSE_FILE}")
        subprocess.run(
            ["docker", "compose", "-f", str(LOCAL_COMPOSE_FILE), "up", "-d"],
            check=True,
        )

        # Wait for containers to start
        time.sleep(2)

        # Check if gateway container is running
        result = subprocess.run(
            ["docker", "ps", "-qf", "name=invariant-gateway"],
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print("The invariant-gateway container failed to start.")
            logs = subprocess.run(
                ["docker", "logs", "invariant-gateway"],
                capture_output=True,
                text=True,
                check=False,
            )
            print("Last 20 lines of logs:")
            print("\n".join(logs.stdout.strip().split("\n")[-20:]))
            return False

        print("Gateway started at http://localhost:8005/api/v1/gateway/")
        print("See http://localhost:8005/api/v1/gateway/docs for API documentation")

        if guardrails_file_path:
            print(f"Using Guardrails File: {guardrails_file_path}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting containers: {e}")
        return False


def down():
    """Stop the Docker containers."""
    try:
        print(f"Stopping containers using docker-compose file: {LOCAL_COMPOSE_FILE}")
        subprocess.run(
            ["docker", "compose", "-f", str(LOCAL_COMPOSE_FILE), "down"],
            check=True,
        )
        print("Containers stopped successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error stopping containers: {e}")
        return False


def logs():
    """Show container logs."""
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(LOCAL_COMPOSE_FILE), "logs", "-f"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error showing logs: {e}")
        return False
    except KeyboardInterrupt:
        print("\nExiting logs view")
        return True


def run_server_command(command, args=None):
    """Run a server command."""
    if args is None:
        args = []

    if command == "build":
        return build()
    elif command == "up":
        # Parse guardrails file from args
        guardrails_file = None
        for arg in args:
            if arg.startswith("--guardrails-file="):
                guardrails_file = arg.split("=", 1)[1]

        return up(guardrails_file)
    elif command == "down":
        return down()
    elif command == "logs":
        return logs()
    else:
        print(f"Unknown server command: {command}")
        print("Available commands: build, up, down, logs")
        return False


def main():
    """Entry point for the Invariant Gateway."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    verb = sys.argv[1]
    if verb == "help":
        print_help()
        return 0

    if "INVARIANT_API_KEY" not in os.environ:
        print("[ERROR] INVARIANT_API_KEY environment variable is not set.")
        mcp_log("[ERROR] INVARIANT_API_KEY environment variable is not set.")
        sys.exit(1)

    if verb == "mcp":
        return asyncio.run(mcp_stdio.execute(sys.argv[2:]))

    if verb == "server":
        if len(sys.argv) < 3:
            print(
                "Error: Missing command for server. Should be one of: build, up, down, logs"
            )
            print_help()
            return 1

        command = sys.argv[2]
        args = sys.argv[3:]
        if not run_server_command(command, args):
            return 1
        return 0

    print(f"[gateway/__main__.py] Unknown action: {verb}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

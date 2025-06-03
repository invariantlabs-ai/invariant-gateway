"""Gateway for MCP (Model Context Protocol) integration with Invariant."""

import asyncio
import json
import os
import platform
import select
import subprocess
import sys

from gateway.common.constants import (
    MCP_METHOD,
    MCP_TOOL_CALL,
    MCP_LIST_TOOLS,
    UTF_8,
)
from gateway.common.mcp_sessions_manager import (
    McpAttributes,
    McpSessionsManager,
)
from gateway.common.mcp_utils import (
    generate_session_id,
    hook_tool_call,
    intercept_response,
    update_mcp_server_in_session_metadata,
    update_session_from_request,
)
from gateway.mcp.log import mcp_log, MCP_LOG_FILE

STATUS_EOF = "eof"
STATUS_DATA = "data"
STATUS_WAIT = "wait"
session_store = McpSessionsManager()


def write_as_utf8_bytes(data: dict) -> bytes:
    """Serializes dict to bytes using UTF-8 encoding."""
    return json.dumps(data).encode(UTF_8) + b"\n"


async def stream_and_forward_stdout(
    session_id: str, mcp_process: subprocess.Popen
) -> None:
    """Read from the mcp_process stdout, apply guardrails and forward to sys.stdout"""
    loop = asyncio.get_event_loop()
    while True:
        if mcp_process.poll() is not None:
            mcp_log(f"[ERROR] MCP process terminated with code: {mcp_process.poll()}")
            break

        line = await loop.run_in_executor(None, mcp_process.stdout.readline)
        if not line:
            break

        try:
            # Process complete JSON lines
            decoded_line = line.decode(UTF_8).strip()
            if not decoded_line:
                continue
            session = session_store.get_session(session_id)
            if session.attributes.verbose:
                mcp_log(f"[INFO] server -> client: {decoded_line}")
            response_body = json.loads(decoded_line)
            update_mcp_server_in_session_metadata(session, response_body)

            intercept_response_result, _ = await intercept_response(
                session_id, session_store, response_body
            )
            # Write and flush immediately
            sys.stdout.buffer.write(write_as_utf8_bytes(intercept_response_result))
            sys.stdout.buffer.flush()
        except Exception as e:  # pylint: disable=broad-except
            mcp_log(f"[ERROR] Error in stream_and_forward_stdout: {str(e)}")
            if line:
                mcp_log(f"[ERROR] Problematic line causing error: {line[:200]}...")


async def stream_and_forward_stderr(
    mcp_process: subprocess.Popen, read_chunk_size: int = 10
) -> None:
    """Read from the mcp_process stderr and write to sys.stderr"""
    loop = asyncio.get_event_loop()

    while True:
        # Read chunks asynchronously
        chunk = await loop.run_in_executor(
            None, lambda: mcp_process.stderr.read(read_chunk_size)
        )

        MCP_LOG_FILE.buffer.write(chunk)
        MCP_LOG_FILE.buffer.flush()


async def _intercept_request(
    session_id: str, mcp_process: subprocess.Popen, line: bytes
) -> None:
    """
    Process a line of input from stdin, decode it and check for guardrails.
    If the request is blocked, it returns a message indicating the block reason
    otherwise it forwards the request to mcp_process stdin.
    """
    session = session_store.get_session(session_id)
    if session.attributes.verbose:
        mcp_log(f"[INFO] client -> server: {line}")

    # Try to decode and parse as JSON to check for tool calls
    try:
        text = line.decode(UTF_8)
        request_body = json.loads(text)
    except json.JSONDecodeError as je:
        mcp_log(f"[ERROR] JSON decode error in run_stdio_input_loop: {str(je)}")
        mcp_log(f"[ERROR] Problematic line: {line[:200]}...")
        return
    update_session_from_request(session, request_body)
    # Refresh guardrails
    await session.load_guardrails()

    hook_tool_call_result = {}
    is_blocked = False
    if request_body.get(MCP_METHOD) == MCP_TOOL_CALL:
        hook_tool_call_result, is_blocked = await hook_tool_call(
            session_id, session_store, request_body
        )
    elif request_body.get(MCP_METHOD) == MCP_LIST_TOOLS:
        hook_tool_call_result, is_blocked = await hook_tool_call(
            session_id=session_id,
            session_store=session_store,
            request_body={
                "id": request_body.get("id"),
                "method": MCP_LIST_TOOLS,
                "params": {"name": MCP_LIST_TOOLS, "arguments": {}},
            },
        )
    if is_blocked:
        sys.stdout.buffer.write(write_as_utf8_bytes(hook_tool_call_result))
        sys.stdout.buffer.flush()
        return
    mcp_process.stdin.write(write_as_utf8_bytes(request_body))
    mcp_process.stdin.flush()


async def wait_for_stdin_input(
    loop: asyncio.AbstractEventLoop, stdin_fd: int
) -> tuple[bytes | None, str]:
    """
    Platform-specific implementation to wait for and read input from stdin.

    Args:
        loop: The asyncio event loop
        stdin_fd: The file descriptor for stdin

    Returns:
        tuple[bytes | None, str]: A tuple containing:
            - The data read from stdin or None
            - Status: 'eof' if EOF detected, 'data' if data available, 'wait' if no data yet
    """
    if platform.system() == "Windows":
        # On Windows, we can't use select for stdin
        # Instead, we'll use a brief sleep and then try to read
        await asyncio.sleep(0.01)
        try:
            chunk = await loop.run_in_executor(None, lambda: os.read(stdin_fd, 4096))
            if not chunk:  # Empty bytes means EOF
                return None, STATUS_EOF
            return chunk, STATUS_DATA
        except (BlockingIOError, OSError):
            # No data available yet
            return None, STATUS_WAIT
    else:
        # On Unix-like systems, use select
        ready, _, _ = await loop.run_in_executor(
            None, lambda: select.select([stdin_fd], [], [], 0.1)
        )

        if not ready:
            # No input available, yield to other tasks
            await asyncio.sleep(0.01)
            return None, STATUS_WAIT

        # Read available data
        chunk = await loop.run_in_executor(None, lambda: os.read(stdin_fd, 4096))
        if not chunk:  # Empty bytes means EOF
            return None, STATUS_EOF
        return chunk, STATUS_DATA


async def run_stdio_input_loop(
    session_id: str,
    mcp_process: subprocess.Popen,
    stdout_task: asyncio.Task,
    stderr_task: asyncio.Task,
) -> None:
    """Handle standard input, intercept call and forward requests to mcp_process stdin."""
    loop = asyncio.get_event_loop()
    stdin_fd = sys.stdin.fileno()
    buffer = b""

    # Set stdin to non-blocking mode
    os.set_blocking(stdin_fd, False)

    try:
        while True:
            # Get input using platform-specific method
            chunk, status = await wait_for_stdin_input(loop, stdin_fd)

            if status == STATUS_EOF:
                # EOF detected, break the loop
                break
            elif status == STATUS_WAIT:
                # No data available yet, continue polling
                continue
            elif status == STATUS_DATA:
                # We got some data, process it
                buffer += chunk

                # Process complete lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue

                    await _intercept_request(session_id, mcp_process, line)
    except (BrokenPipeError, KeyboardInterrupt):
        # Broken pipe = client disappeared, just start shutdown
        mcp_log("Client disconnected or keyboard interrupt")
    finally:
        # Close stdin
        if mcp_process.stdin:
            mcp_process.stdin.close()

        # Process any remaining data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line:
                await _intercept_request(session_id, mcp_process, line)

        # Terminate process if needed
        if mcp_process.poll() is None:
            mcp_process.terminate()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, mcp_process.wait), timeout=2
                )
            except asyncio.TimeoutError:
                mcp_process.kill()

        # Cancel I/O tasks
        stdout_task.cancel()
        stderr_task.cancel()

        # Final flush
        sys.stdout.flush()


def split_args(args: list[str] = None) -> tuple[list[str], list[str]]:
    """
    Splits CLI arguments into two parts:
    1. Arguments intended for the MCP gateway (everything before `--exec`)
    2. Arguments for the underlying MCP server (everything after `--exec`)

    Parameters:
        args (list[str]): The list of CLI arguments.

    Returns:
        Tuple[list[str], list[str]]: A tuple containing (mcp_gateway_args, mcp_server_command_args)
    """
    if not args:
        mcp_log("[ERROR] No arguments provided.")
        sys.exit(1)

    try:
        exec_index = args.index("--exec")
    except ValueError:
        mcp_log("[ERROR] '--exec' flag not found in arguments.")
        sys.exit(1)

    mcp_gateway_args = args[:exec_index]
    mcp_server_command_args = args[exec_index + 1 :]

    if not mcp_server_command_args:
        mcp_log("[ERROR] No arguments provided after '--exec'.")
        sys.exit(1)

    return mcp_gateway_args, mcp_server_command_args


async def execute(args: list[str] = None):
    """Main function to execute the MCP gateway."""
    if "INVARIANT_API_KEY" not in os.environ:
        mcp_log("[ERROR] INVARIANT_API_KEY environment variable is not set.")
        sys.exit(1)

    mcp_log("[INFO] Running with Python version:", sys.version)

    mcp_gateway_args, mcp_server_command_args = split_args(args)
    session_id = generate_session_id()
    await session_store.initialize_session(
        session_id,
        McpAttributes.from_cli_args(mcp_gateway_args),
    )

    mcp_process = subprocess.Popen(
        mcp_server_command_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Start async tasks for stdout and stderr
    stdout_task = asyncio.create_task(
        stream_and_forward_stdout(session_id, mcp_process)
    )
    stderr_task = asyncio.create_task(stream_and_forward_stderr(mcp_process))

    # Handle forwarding stdin and intercept tool calls
    await run_stdio_input_loop(session_id, mcp_process, stdout_task, stderr_task)

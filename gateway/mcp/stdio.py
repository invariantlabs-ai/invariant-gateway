"""Gateway for MCP (Model Context Protocol) integration with Invariant."""

import asyncio
import json
import os
import platform
import select
import subprocess
import sys

from gateway.mcp.constants import UTF_8
from gateway.mcp.log import mcp_log, MCP_LOG_FILE
from gateway.mcp.mcp_sessions_manager import (
    McpAttributes,
    McpSessionsManager,
)
from gateway.mcp.mcp_transport_base import McpTransportBase

STATUS_EOF = "eof"
STATUS_DATA = "data"
STATUS_WAIT = "wait"
mcp_sessions_manager = McpSessionsManager()


class StdioTransport(McpTransportBase):
    """
    STDIO transport implementation for MCP communication.
    Handles subprocess-based communication with stdin/stdout/stderr.
    """

    def __init__(self, session_store: McpSessionsManager):
        super().__init__(session_store)
        self.mcp_process: subprocess.Popen = None

    async def initialize_session(self, **kwargs) -> str:
        """Initialize session for stdio transport."""
        session_attributes: McpAttributes = kwargs.get("session_attributes")
        session_id = self.generate_session_id()
        await self.session_store.initialize_session(session_id, session_attributes)
        mcp_log(f"Created stdio session with ID: {session_id}")
        return session_id

    def start_mcp_process(self, mcp_server_command_args: list) -> subprocess.Popen:
        """Start the MCP server subprocess."""
        self.mcp_process = subprocess.Popen(
            mcp_server_command_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        mcp_log(f"Started MCP process with PID: {self.mcp_process.pid}")
        return self.mcp_process

    async def handle_communication(self, **kwargs) -> None:
        """Handle stdio communication loop."""
        session_id: str = kwargs.get("session_id")
        mcp_process: subprocess.Popen = kwargs.get("mcp_process")
        if not session_id or not mcp_process:
            raise ValueError(
                "session_id and mcp_process are required for stdio transport"
            )

        self.mcp_process = mcp_process

        # Start async tasks for stdout and stderr
        stdout_task = asyncio.create_task(self._stream_and_forward_stdout(session_id))
        stderr_task = asyncio.create_task(self._stream_and_forward_stderr())

        try:
            # Handle stdin input loop
            await self._run_stdio_input_loop(session_id)
        finally:
            # Cleanup
            if self.mcp_process and self.mcp_process.stdin:
                self.mcp_process.stdin.close()

            # Terminate process if needed
            if self.mcp_process and self.mcp_process.poll() is None:
                self.mcp_process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.mcp_process.wait
                        ),
                        timeout=2,
                    )
                except asyncio.TimeoutError:
                    self.mcp_process.kill()

            # Cancel I/O tasks
            stdout_task.cancel()
            stderr_task.cancel()

            # Final flush
            sys.stdout.flush()

    async def _stream_and_forward_stdout(self, session_id: str) -> None:
        """Read from MCP process stdout, apply guardrails and forward to sys.stdout."""
        loop = asyncio.get_event_loop()

        while True:
            if self.mcp_process.poll() is not None:
                mcp_log(
                    f"[ERROR] MCP process terminated with code: {self.mcp_process.poll()}"
                )
                break

            line = await loop.run_in_executor(None, self.mcp_process.stdout.readline)
            if not line:
                break

            try:
                decoded_line = line.decode(UTF_8).strip()
                if not decoded_line:
                    continue

                session = self.session_store.get_session(session_id)
                if session.attributes.verbose:
                    mcp_log(f"[INFO] server -> client: {decoded_line}")

                response_body = json.loads(decoded_line)
                processed_response, _ = await self.process_incoming_response(
                    session_id, response_body
                )

                sys.stdout.buffer.write(self._serialize_to_bytes(processed_response))
                sys.stdout.buffer.flush()
            except Exception as e:  # pylint: disable=broad-except
                mcp_log(f"[ERROR] Error in _stream_and_forward_stdout: {str(e)}")
                if line:
                    mcp_log(f"[ERROR] Problematic line: {line[:200]}...")

    async def _stream_and_forward_stderr(self) -> None:
        """Read from MCP process stderr and write to log file."""
        loop = asyncio.get_event_loop()

        while True:
            chunk = await loop.run_in_executor(
                None, lambda: self.mcp_process.stderr.read(10)
            )
            if not chunk:
                break
            MCP_LOG_FILE.buffer.write(chunk)
            MCP_LOG_FILE.buffer.flush()

    async def _run_stdio_input_loop(self, session_id: str) -> None:
        """Handle standard input, intercept calls and forward requests to MCP process stdin."""
        loop = asyncio.get_event_loop()
        stdin_fd = sys.stdin.fileno()
        buffer = b""

        # Set stdin to non-blocking mode
        os.set_blocking(stdin_fd, False)

        try:
            while True:
                # Get input using platform-specific method
                chunk, status = await self._wait_for_stdin_input(loop, stdin_fd)

                if status == STATUS_EOF:
                    break
                elif status == STATUS_WAIT:
                    continue
                elif status == STATUS_DATA:
                    buffer += chunk

                    # Process complete lines
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line:
                            continue

                        await self._process_stdin_line(session_id, line)

        except (BrokenPipeError, KeyboardInterrupt):
            mcp_log("Client disconnected or keyboard interrupt")
        finally:
            # Process any remaining data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line:
                    await self._process_stdin_line(session_id, line)

    async def _process_stdin_line(self, session_id: str, line: bytes) -> None:
        """Process a line of input from stdin."""
        session = self.session_store.get_session(session_id)
        if session.attributes.verbose:
            mcp_log(f"[INFO] client -> server: {line}")

        try:
            text = line.decode(UTF_8)
            request_body = json.loads(text)
        except json.JSONDecodeError as je:
            mcp_log(f"[ERROR] JSON decode error: {str(je)}")
            mcp_log(f"[ERROR] Problematic line: {line[:200]}...")
            return

        processed_request, is_blocked = await self.process_outgoing_request(
            session_id, request_body
        )

        if is_blocked:
            sys.stdout.buffer.write(self._serialize_to_bytes(processed_request))
            sys.stdout.buffer.flush()
            return
        self.mcp_process.stdin.write(self._serialize_to_bytes(request_body))
        self.mcp_process.stdin.flush()

    async def _wait_for_stdin_input(
        self, loop: asyncio.AbstractEventLoop, stdin_fd: int
    ) -> tuple[bytes | None, str]:
        """Platform-specific implementation to wait for and read input from stdin."""
        if platform.system() == "Windows":
            await asyncio.sleep(0.01)
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: os.read(stdin_fd, 4096)
                )
                if not chunk:
                    return None, STATUS_EOF
                return chunk, STATUS_DATA
            except (BlockingIOError, OSError):
                return None, STATUS_WAIT
        else:
            # Unix-like systems
            ready, _, _ = await loop.run_in_executor(
                None, lambda: select.select([stdin_fd], [], [], 0.1)
            )

            if not ready:
                await asyncio.sleep(0.01)
                return None, STATUS_WAIT

            chunk = await loop.run_in_executor(None, lambda: os.read(stdin_fd, 4096))
            if not chunk:
                return None, STATUS_EOF
            return chunk, STATUS_DATA

    def _serialize_to_bytes(self, data: dict) -> bytes:
        """Serialize dict to bytes using UTF-8 encoding."""
        return json.dumps(data).encode(UTF_8) + b"\n"


async def create_stdio_transport_and_execute(
    session_store: McpSessionsManager,
    session_attributes: McpAttributes,
    mcp_server_command_args: list,
) -> None:
    """Integration function for stdio execution."""
    stdio_transport = StdioTransport(session_store=session_store)

    session_id = await stdio_transport.initialize_session(
        session_attributes=session_attributes
    )

    await stdio_transport.handle_communication(
        session_id=session_id,
        mcp_process=stdio_transport.start_mcp_process(mcp_server_command_args),
    )


def split_args(args: list[str] | None = None) -> tuple[list[str], list[str]]:
    """
    Splits CLI arguments into two parts:
    1. Arguments intended for the MCP gateway (everything before `--exec`)
    2. Arguments for the underlying MCP server (everything after `--exec`)
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
    """Main function to execute the MCP gateway using transport strategy."""
    if "INVARIANT_API_KEY" not in os.environ:
        mcp_log("[ERROR] INVARIANT_API_KEY environment variable is not set.")
        sys.exit(1)

    mcp_log("[INFO] Running with Python version:", sys.version)

    # Parse arguments
    mcp_gateway_args, mcp_server_command_args = split_args(args)

    # Create session store and attributes
    session_attributes = McpAttributes.from_cli_args(mcp_gateway_args)

    # Use stdio transport strategy
    await create_stdio_transport_and_execute(
        session_store=mcp_sessions_manager,
        session_attributes=session_attributes,
        mcp_server_command_args=mcp_server_command_args,
    )

"""Gateway for MCP (Model Context Protocol) integration with Invariant."""

import sys
import subprocess
import json
import os
import select
import asyncio
import platform

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.append_messages import AppendMessagesRequest
from invariant_sdk.types.push_traces import PushTracesRequest

from gateway.common.constants import (
    INVARIANT_GUARDRAILS_BLOCKED_MESSAGE,
    INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE,
    MCP_METHOD,
    MCP_CLIENT_INFO,
    MCP_PARAMS,
    MCP_SERVER_INFO,
    MCP_TOOL_CALL,
    MCP_LIST_TOOLS,
)
from gateway.common.guardrails import GuardrailAction
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import create_annotations_from_guardrails_errors
from gateway.integrations.guardrails import check_guardrails
from gateway.mcp.log import mcp_log, MCP_LOG_FILE, format_errors_in_response
from gateway.mcp.mcp_context import McpContext
from gateway.mcp.task_utils import run_task_in_background, run_task_sync
import getpass
import socket

UTF_8_ENCODING = "utf-8"
DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


def user_and_host() -> str:
    username = getpass.getuser()
    hostname = socket.gethostname()

    return f"{username}@{hostname}"


def session_metadata(ctx: McpContext) -> dict:
    return {
        "session_id": ctx.local_session_id,
        "system_user": user_and_host(),
        "mcp_client": ctx.mcp_client_name,
        "mcp_server": ctx.mcp_server_name,
        "tools": ctx.tools,
        **(ctx.extra_metadata or {}),
    }


def write_as_utf8_bytes(data: dict) -> bytes:
    """Serializes dict to bytes using UTF-8 encoding."""
    return json.dumps(data).encode(UTF_8_ENCODING) + b"\n"


def deduplicate_annotations(ctx: McpContext, new_annotations: list) -> list:
    """Deduplicate new_annotations using the annotations in the context."""
    deduped_annotations = []
    for annotation in new_annotations:
        # Check if an annotation with the same content and address exists in ctx.annotations
        # TODO: Rely on the __eq__ method of the AnnotationCreate class directly via not in
        # to remove duplicates instead of using a custom logic.
        # This is a temporary solution until the Invariant SDK is updated.
        is_duplicate = False
        for ctx_annotation in ctx.annotations:
            if (
                annotation.content == ctx_annotation.content
                and annotation.address == ctx_annotation.address
                and annotation.extra_metadata == ctx_annotation.extra_metadata
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            deduped_annotations.append(annotation)

    return deduped_annotations


def check_if_new_errors(ctx: McpContext, guardrails_result: dict) -> bool:
    """Checks if there are new errors in the guardrails result."""
    annotations = create_annotations_from_guardrails_errors(
        guardrails_result.get("errors", [])
    )
    for annotation in annotations:
        if annotation not in ctx.annotations:
            return True
    return False


async def get_guardrails_check_result(
    ctx: McpContext,
    message: dict,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> dict:
    """
    Check against guardrails of type action in an async manner.
    """
    # Skip if no guardrails are configured for this action
    if not (
        (ctx.guardrails.blocking_guardrails and action == GuardrailAction.BLOCK)
        or (ctx.guardrails.logging_guardrails and action == GuardrailAction.LOG)
    ):
        return {}

    # Prepare context and select appropriate guardrails
    context = RequestContext.create(
        request_json={},
        dataset_name=ctx.explorer_dataset,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        guardrails=ctx.guardrails,
    )

    guardrails_to_check = (
        ctx.guardrails.blocking_guardrails
        if action == GuardrailAction.BLOCK
        else ctx.guardrails.logging_guardrails
    )

    # Run check_guardrails asynchronously
    return await check_guardrails(
        messages=ctx.trace + [message],
        guardrails=guardrails_to_check,
        context=context,
    )


async def append_and_push_trace(
    ctx: McpContext, message: dict, guardrails_result: dict
) -> None:
    """
    Append a message to the trace if it exists or create a new one
    and push it to the Invariant Explorer.
    """
    annotations = []
    if guardrails_result and guardrails_result.get("errors", []):
        annotations = create_annotations_from_guardrails_errors(
            guardrails_result["errors"]
        )

    if ctx.guardrails.logging_guardrails:
        logging_guardrails_check_result = await get_guardrails_check_result(
            ctx, message, action=GuardrailAction.LOG
        )
        if logging_guardrails_check_result and logging_guardrails_check_result.get(
            "errors", []
        ):
            annotations.extend(
                create_annotations_from_guardrails_errors(
                    logging_guardrails_check_result["errors"]
                )
            )
    deduplicated_annotations = deduplicate_annotations(ctx, annotations)

    try:
        # If the trace_id is None, create a new trace with the messages.
        # Otherwise, append the message to the existing trace.
        client = AsyncClient(
            api_url=os.getenv("INVARIANT_API_URL", DEFAULT_API_URL),
        )

        if ctx.trace_id is None:
            ctx.trace.append(message)

            # default metadata
            metadata = {"source": "mcp"}
            # include MCP session metadata
            metadata.update(session_metadata(ctx))

            response = await client.push_trace(
                PushTracesRequest(
                    messages=[ctx.trace],
                    dataset=ctx.explorer_dataset,
                    metadata=[metadata],
                    annotations=[deduplicated_annotations],
                )
            )
            ctx.trace_id = response.id[0]
            ctx.last_trace_length = len(ctx.trace)
            ctx.annotations.extend(deduplicated_annotations)
        else:
            ctx.trace.append(message)
            response = await client.append_messages(
                AppendMessagesRequest(
                    trace_id=ctx.trace_id,
                    messages=ctx.trace[ctx.last_trace_length :],
                    annotations=deduplicated_annotations,
                )
            )
            ctx.last_trace_length = len(ctx.trace)
            ctx.annotations.extend(deduplicated_annotations)
    except Exception as e:
        mcp_log("[ERROR] Error pushing trace in append_and_push_trace:", e)


async def get_guardrails_check_result(
    ctx: McpContext,
    message: dict,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> dict:
    """
    Check against guardrails of type action.
    Works in both sync and async contexts by always using a dedicated thread.
    """
    # Skip if no guardrails are configured for this action
    if not (
        (ctx.guardrails.blocking_guardrails and action == GuardrailAction.BLOCK)
        or (ctx.guardrails.logging_guardrails and action == GuardrailAction.LOG)
    ):
        return {}

    # Prepare context and select appropriate guardrails
    context = RequestContext.create(
        request_json={},
        dataset_name=ctx.explorer_dataset,
        invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
        guardrails=ctx.guardrails,
        guardrails_parameters={"metadata": session_metadata(ctx), "action": action},
    )

    guardrails_to_check = (
        ctx.guardrails.blocking_guardrails
        if action == GuardrailAction.BLOCK
        else ctx.guardrails.logging_guardrails
    )

    return run_task_sync(
        check_guardrails,
        messages=ctx.trace + [message],
        guardrails=guardrails_to_check,
        context=context,
    )


def json_rpc_error_response(
    id_value: str | int, error_message: str, response_type: str = "error"
) -> dict:
    """
    Create a JSON-RPC error response with either error object or content format.

    Args:
        id_value: The ID of the JSON-RPC request
        error_message: The error message to include
        response_type: Either "error" or "content" to determine response format

    Returns:
        A properly formatted JSON-RPC response dictionary
    """
    base_response = {
        "jsonrpc": "2.0",
        "id": id_value,
    }

    if response_type == "error":
        base_response["error"] = {
            "code": -32600,
            "message": error_message,
        }
    else:
        base_response["result"] = {
            "content": [
                {
                    "type": "text",
                    "text": error_message,
                }
            ]
        }

    return base_response


async def hook_tool_call(ctx: McpContext, request: dict) -> tuple[dict, bool]:
    """
    Hook function to intercept tool calls.

    If the request is blocked, it returns a tuple with a message explaining the block
    and a flag indicating the request was blocked.
    Otherwise it returns the original request and a flag indicating it was not blocked.
    """
    tool_call = {
        "id": f"call_{request.get('id')}",
        "type": "function",
        "function": {
            "name": request["params"]["name"],
            "arguments": request["params"]["arguments"],
        },
    }

    message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}

    # Check for blocking guardrails
    guardrailing_result = await get_guardrails_check_result(
        ctx, message, action=GuardrailAction.BLOCK
    )

    # If the request is blocked, return a message indicating the block reason.
    if (
        guardrailing_result
        and guardrailing_result.get("errors", [])
        and check_if_new_errors(ctx, guardrailing_result)
    ):
        if ctx.push_explorer:
            await append_and_push_trace(ctx, message, guardrailing_result)
        else:
            ctx.trace.append(message)

        return json_rpc_error_response(
            request.get("id"),
            INVARIANT_GUARDRAILS_BLOCKED_MESSAGE % guardrailing_result["errors"],
            response_type=ctx.failure_response_format,
        ), True

    # Add the message to the trace
    ctx.trace.append(message)
    return request, False


async def hook_tool_result(ctx: McpContext, result: dict) -> dict:
    """
    Hook function to intercept tool results.
    Returns the potentially modified result.
    """
    method = ctx.id_to_method_mapping.get(result.get("id"))
    call_id = f"call_{result.get('id')}"

    # Safely handle result object
    result_obj = result.get("result", {})
    if isinstance(result_obj, dict) and MCP_SERVER_INFO in result_obj:
        ctx.mcp_server_name = result_obj.get(MCP_SERVER_INFO, {}).get("name", "")

    if method is None:
        return result
    elif method == MCP_TOOL_CALL:
        message = {
            "role": "tool",
            "content": result_obj.get("content"),
            "error": result_obj.get("error"),
            "tool_call_id": call_id
        }
        # Check for blocking guardrails
        guardrailing_result = await get_guardrails_check_result(
            ctx, message, action=GuardrailAction.BLOCK
        )

        if guardrailing_result and guardrailing_result.get("errors", []):
            result = json_rpc_error_response(
                result.get("id"),
                INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                % format_errors_in_response(guardrailing_result["errors"]),
                response_type=ctx.failure_response_format,  # Using content type as that's what the original code used
            )

        if ctx.push_explorer:
            await append_and_push_trace(ctx, message, guardrailing_result)
        else:
            ctx.trace.append(message)

        return result
    elif method == MCP_LIST_TOOLS:
        ctx.tools = result_obj.get("tools")
        message = {
            "role": "tool",
            "content": json.dumps(result.get("result").get("tools")),
            "tool_call_id": call_id,
        }
        # next validate it with guardrails
        guardrailing_result = await get_guardrails_check_result(
            ctx, message, action=GuardrailAction.BLOCK
        )
        if guardrailing_result and guardrailing_result.get("errors", []):
            result["result"]["tools"] = [
                {
                    "name": "blocked_" + tool["name"],
                    "description": INVARIANT_GUARDRAILS_BLOCKED_TOOLS_MESSAGE
                    % format_errors_in_response(guardrailing_result["errors"]),
                    # no parameters
                    "inputSchema": {
                        "properties": {},
                        "required": [],
                        "title": "invariant_mcp_server_blockedArguments",
                        "type": "object",
                    },
                    "annotations": {
                        "title": "This tool was blocked by security guardrails.",
                    },
                }
                for tool in result["result"]["tools"]
            ]

        # add it to the session trace (and run logging guardrails)
        if ctx.push_explorer:
            await append_and_push_trace(ctx, message, guardrailing_result)
        else:
            ctx.trace.append(message)

        return result
    else:
        return result


async def stream_and_forward_stdout(
    mcp_process: subprocess.Popen, ctx: McpContext
) -> None:
    """Read from the mcp_process stdout, apply guardrails and forward to sys.stdout"""
    loop = asyncio.get_event_loop()

    while True:
        line = await loop.run_in_executor(None, mcp_process.stdout.readline)
        if not line:
            break

        try:
            # Process complete JSON lines
            line_str = line.decode(UTF_8_ENCODING).strip()
            if not line_str:
                continue

            if ctx.verbose:
                mcp_log(f"[INFO] server -> client: {line_str}")

            parsed_json = json.loads(line_str)
            processed_json = await hook_tool_result(ctx, parsed_json)

            # Write and flush immediately
            sys.stdout.buffer.write(write_as_utf8_bytes(processed_json))
            sys.stdout.buffer.flush()

        except Exception as e:
            import traceback
            mcp_log(traceback.format_exc())
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


async def process_line(
    ctx: McpContext, mcp_process: subprocess.Popen, line: bytes
) -> None:
    if ctx.verbose:
        mcp_log(f"[INFO] client -> server: {line}")

    # Try to decode and parse as JSON to check for tool calls
    try:
        text = line.decode(UTF_8_ENCODING)
        parsed_json = json.loads(text)
    except json.JSONDecodeError as je:
        mcp_log(f"[ERROR] JSON decode error in run_stdio_input_loop: {str(je)}")
        mcp_log(f"[ERROR] Problematic line: {line[:200]}...")
        return

    if parsed_json.get(MCP_METHOD) is not None:
        ctx.id_to_method_mapping[parsed_json.get("id")] = parsed_json.get(
            MCP_METHOD
        )
    if "params" in parsed_json and "clientInfo" in parsed_json.get("params"):
        ctx.mcp_client_name = (
            parsed_json.get("params").get("clientInfo").get("name", "")
        )

    # Check if this is a tool call request
    if parsed_json.get(MCP_METHOD) == MCP_TOOL_CALL:
        # Refresh guardrails
        run_task_sync(ctx.load_guardrails)

        # Intercept and potentially block modify the request
        hook_tool_call_result, is_blocked = await hook_tool_call(ctx, parsed_json)
        if not is_blocked:
            # If blocked, hook_tool_call_result contains the original request.
            # Forward the request to the MCP process.
            # It will handle the request and return a response.
            mcp_process.stdin.write(write_as_utf8_bytes(hook_tool_call_result))
            mcp_process.stdin.flush()
        else:
            # If blocked, hook_tool_call_result contains the block message.
            # Forward the block message result back to the caller.
            # The original request is not passed to the MCP process.
            sys.stdout.buffer.write(write_as_utf8_bytes(hook_tool_call_result))
            sys.stdout.buffer.flush()
        return
    else:
        # pass through the request to the MCP process

        # for list_tools, extend the trace by a tool call
        if parsed_json.get(MCP_METHOD) == MCP_LIST_TOOLS:
            # Refresh guardrails
            run_task_sync(ctx.load_guardrails)

            # mcp_message_{}
            ctx.trace.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{parsed_json.get('id')}",
                            "type": "function",
                            "function": {
                                "name": "tools/list",
                                "arguments": {},
                            },
                        }
                    ],
                }
            )
        mcp_process.stdin.write(write_as_utf8_bytes(parsed_json))
        mcp_process.stdin.flush()


async def wait_for_stdin_input(loop: asyncio.AbstractEventLoop, stdin_fd: int) -> tuple[bytes | None, str]:
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
                return None, 'eof'
            return chunk, 'data'
        except (BlockingIOError, OSError):
            # No data available yet
            return None, 'wait'
    else:
        # On Unix-like systems, use select
        ready, _, _ = await loop.run_in_executor(
            None, lambda: select.select([stdin_fd], [], [], 0.1)
        )

        if not ready:
            # No input available, yield to other tasks
            await asyncio.sleep(0.01)
            return None, 'wait'

        # Read available data
        chunk = await loop.run_in_executor(None, lambda: os.read(stdin_fd, 4096))
        if not chunk:  # Empty bytes means EOF
            return None, 'eof'
        return chunk, 'data'


async def run_stdio_input_loop(
    ctx: McpContext,
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
            
            if status == 'eof':
                # EOF detected, break the loop
                break
            elif status == 'wait':
                # No data available yet, continue polling
                continue
            elif status == 'data':
                # We got some data, process it
                buffer += chunk

                # Process complete lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue

                    await process_line(ctx, mcp_process, line)
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
                await process_line(ctx, mcp_process, line)

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
    ctx = McpContext(mcp_gateway_args)

    mcp_process = subprocess.Popen(
        mcp_server_command_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Start async tasks for stdout and stderr
    stdout_task = asyncio.create_task(stream_and_forward_stdout(mcp_process, ctx))
    stderr_task = asyncio.create_task(stream_and_forward_stderr(mcp_process))

    # Handle forwarding stdin and intercept tool calls
    await run_stdio_input_loop(ctx, mcp_process, stdout_task, stderr_task)

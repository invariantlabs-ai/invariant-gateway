"""Gateway for MCP (Model Context Protocol) integration with Invariant."""

import argparse
import asyncio
import sys
import subprocess
import json
import os
import threading
import signal

from builtins import print as builtins_print
from contextlib import redirect_stdout

from gateway.common.request_context import RequestContext
from gateway.integrations.guardrails import check_guardrails
from gateway.integrations.explorer import (
    fetch_guardrails_from_explorer,
)
from gateway.mcp.mcp_context import McpContext


def custom_print(ctx, *args, **kwargs):
    """Custom print function to redirect output to log_out."""
    builtins_print(*args, **kwargs, file=ctx.log_out, flush=True)


def append_and_push_trace(ctx, message):
    """
    Append a message to the trace if it exists or create a new one
    and push it to the Invariant Explorer.
    """
    try:
        if ctx.trace_id is None:
            ctx.trace.append(message)
            response = ctx.client.create_request_and_push_trace(
                messages=[ctx.trace],
                dataset=ctx.explorer_dataset,
                metadata=[{"source": "mcp", "tools": ctx.tools}],
            )
            ctx.trace_id = response.id[0]
            ctx.last_trace_length = len(ctx.trace)
        else:
            ctx.trace.append(message)
            ctx.client.create_request_and_append_messages(
                trace_id=ctx.trace_id, messages=ctx.trace[ctx.last_trace_length :]
            )
            ctx.last_trace_length = len(ctx.trace)
    except Exception as e:
        custom_print(ctx, "Error pushing trace:", e)


def fetch_guardrails(ctx, dataset):
    """Fetch guardrails from the Invariant Explorer."""
    # Use async fetch_guardrails_from_explorer in a thread
    return asyncio.run(
        fetch_guardrails_from_explorer(
            dataset, "Bearer " + os.getenv("INVARIANT_API_KEY")
        )
    )


def check_blocking_guardrails(ctx, message, request):
    """Check against blocking guardrails."""
    try:
        guardrails = fetch_guardrails(ctx, ctx.explorer_dataset)

        custom_print(ctx, "Here are the guardrails: ", guardrails)

        context = RequestContext.create(
            request_json=request,
            dataset_name=ctx.explorer_dataset,
            invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
            guardrails=guardrails,
        )

        if guardrails.blocking_guardrails:
            with redirect_stdout(ctx.log_out):
                return asyncio.run(
                    check_guardrails(
                        messages=ctx.trace + [message],
                        guardrails=guardrails.blocking_guardrails,
                        context=context,
                    )
                )
        else:
            return {}
    except Exception as e:
        custom_print(ctx, "Error checking blocking guardrails:", e)


def hook_tool_call(ctx, request):
    """
    Hook function to intercept tool calls.
    Modify this function to change behavior for tool calls.
    Returns the potentially modified request.
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
    result = check_blocking_guardrails(ctx, message, request)
    append_and_push_trace(ctx, message)
    return request


def hook_tool_result(ctx, result):
    """
    Hook function to intercept tool results.
    Modify this function to change behavior for tool results.
    Returns the potentially modified result.
    """
    method = ctx.id_to_method_mapping.get(result.get("id"))
    call_id = f"call_{result.get('id')}"

    if method is None:
        return result
    elif method == "tools/call":
        message = {
            "role": "tool",
            "content": {"result": result.get("result").get("content")},
            "error": result.get("result").get("error"),
            "tool_call_id": call_id,
        }

        # Check for blocking guardrails
        guardrailing_result = check_blocking_guardrails(ctx, message, result)

        if guardrailing_result and guardrailing_result.get("errors", []):
            result["result"]["content"] = [
                {
                    "type": "text",
                    "text": "[Invariant] Your MCP tool call was blocked for security reasons. Do not attempt to circumvent this block, rather explain to the user based on the following output what went wrong: \n"
                    + json.dumps(guardrailing_result["errors"]),
                }
            ]

        append_and_push_trace(ctx, message)
        return result
    elif method == "tools/list":
        ctx.tools = result.get("result").get("tools")
        return result
    else:
        return result


def forward_stdout(process, ctx, buffer_size=1):
    """Read from the process stdout, parse JSON chunks, and forward to sys.stdout"""
    buffer = b""

    while True:
        chunk = process.stdout.read(buffer_size)
        if not chunk:
            break
        buffer += chunk

        try:
            # Try parsing full JSON object from buffer
            text = buffer.decode("utf-8")
            obj = json.loads(text)

            obj = hook_tool_result(ctx, obj)
            # clear the buffer
            buffer = b""

            # Forward the original JSON to stdout
            json_output = json.dumps(obj).encode("utf-8") + b"\n"
            sys.stdout.buffer.write(json_output)
            sys.stdout.buffer.flush()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Wait for more data
            continue


def forward_stderr(process, ctx, buffer_size=1):
    """Read from the process stderr and write to sys.stderr"""
    for line in iter(lambda: process.stderr.read(buffer_size), b""):
        ctx.log_out.buffer.write(line)
        ctx.log_out.buffer.flush()


def execute(args=None):
    """Main function to execute the MCP gateway."""
    if "INVARIANT_API_KEY" not in os.environ:
        print("[ERROR] INVARIANT_API_KEY environment variable is not set.")
        sys.exit(1)

    # Split args at the "--exec" boundary
    if args and "--exec" in args:
        exec_index = args.index("--exec")
        pre_exec_args = args[:exec_index]
        post_exec_args = args[exec_index + 1 :]
    else:
        pre_exec_args = args or []
        post_exec_args = []

    if not post_exec_args:
        print("[ERROR] No command provided after --exec.")
        sys.exit(1)

    # Parse pre-exec args using argparse
    parser = argparse.ArgumentParser(description="MCP Gateway")
    parser.add_argument("--directory", help="Working directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    config = parser.parse_args(pre_exec_args)
    # Initialize the singleton context using config
    ctx = McpContext()

    # Can now use post_exec_args as your cmd
    cmd = post_exec_args

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # No buffering
    )

    # Start threads to forward stdout and stderr
    stdout_thread = threading.Thread(
        target=forward_stdout, args=(process, ctx), daemon=True
    )
    stderr_thread = threading.Thread(
        target=forward_stderr, args=(process, ctx), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    # Handle forwarding stdin and intercept tool calls
    try:
        current_chunk = b""

        while True:
            data = sys.stdin.buffer.read(1)
            current_chunk += data

            if not data:
                break

            # Try to decode and parse as JSON to check for tool calls
            try:
                text = current_chunk.decode("utf-8")
                obj = json.loads(text)
                # clear the current chunk
                current_chunk = b""

                if obj.get("method") is not None:
                    ctx.id_to_method_mapping[obj.get("id")] = obj.get("method")

                # Check if this is a tool call request
                if obj.get("method") == "tools/call":
                    # Intercept and potentially modify the request
                    obj = hook_tool_call(ctx, obj)
                    # Convert back to bytes
                    data = json.dumps(obj).encode("utf-8")

                    # Forward to the process
                    process.stdin.write(data + b"\n")
                    process.stdin.flush()
                    continue
                else:
                    process.stdin.write(json.dumps(obj).encode("utf-8") + b"\n")
                    process.stdin.flush()
                    continue
            except Exception:
                # Not a complete or valid JSON, just pass through
                pass

    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        process.terminate()

    # Wait for process to terminate
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    ctx = McpContext()
    custom_print(ctx, f"Received signal {sig}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    execute(sys.argv)

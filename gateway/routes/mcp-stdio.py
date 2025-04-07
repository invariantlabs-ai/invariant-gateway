#!/usr/bin/env python3
import asyncio
import sys
import subprocess
import json
import os
import threading
import signal
from invariant_sdk.client import Client

from contextlib import redirect_stdout

# ensure ../ is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.request_context import RequestContext
from integrations.guardrails import check_guardrails
from common.guardrails import GuardrailRuleSet
from integrations.explorer import (
    fetch_guardrails_from_explorer,
)

# requires the 'INVARIANT_API_KEY' environment variable to be set
client = Client()

# trace state (continously expanded on)
EXPLORER_DATASET = "mcp-capture"
TRACE = []
TOOLS = []
trace_id = None
last_trace_length = 0

# guardrailing state
GUARDRAILS: GuardrailRuleSet | None = None

# maps JSON RPC IDs to method names
id_to_method_mapping = {}

# set stderr to be log.txt in the ~/.invariant/mcp.log
os.makedirs(os.path.join(os.path.expanduser("~"), ".invariant"), exist_ok=True)
LOG_OUT = open(
    os.path.join(os.path.expanduser("~"), ".invariant", "mcp.log"),
    "a",
    buffering=1,
)
sys.stderr = LOG_OUT


def print(*args, **kwargs):
    from builtins import print as builtins_print

    builtins_print(*args, **kwargs, file=LOG_OUT, flush=True)


def append_and_push_trace(message):
    global trace_id, TRACE, last_trace_length, tool_call_ids

    try:
        if trace_id is None:
            TRACE.append(message)
            response = client.create_request_and_push_trace(
                messages=[TRACE],
                dataset="mcp-capture",
                metadata=[{"source": "mcp", "tools": TOOLS}],
            )
            trace_id = response.id[0]
            last_trace_length = len(TRACE)
        else:
            TRACE.append(message)
            client.create_request_and_append_messages(
                trace_id=trace_id, messages=TRACE[last_trace_length:]
            )
            last_trace_length = len(TRACE)
    except Exception as e:
        import traceback

        print(traceback.format_exc())


def check_blocking_guardrails(message, request):
    try:
        guardrails = fetch_guardrails(EXPLORER_DATASET)

        context = RequestContext.create(
            request_json=request,
            dataset_name=EXPLORER_DATASET,
            invariant_authorization="Bearer " + os.getenv("INVARIANT_API_KEY"),
            guardrails=guardrails,
        )

        with redirect_stdout(LOG_OUT):
            return asyncio.run(
                check_guardrails(
                    messages=TRACE + [message],
                    guardrails=guardrails.blocking_guardrails,
                    context=context,
                )
            )
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise e


def hook_tool_call(request):
    """
    Hook function to intercept tool calls.
    Modify this function to change behavior for tool calls.
    Returns the potentially modified request.
    """
    global trace_id, TRACE

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
    result = check_blocking_guardrails(message, request)
    print("Guardrails check tool call result:", result)

    append_and_push_trace(message)

    return request


def hook_tool_result(result):
    """
    Hook function to intercept tool results.
    Modify this function to change behavior for tool results.
    Returns the potentially modified result.
    """
    global TOOLS

    method = id_to_method_mapping.get(result.get("id"))
    call_id = f"call_{result.get('id')}"

    if method is None:
        return result
    elif method == "tools/call":
        message = {
            "role": "tool",
            "content": json.dumps(result.get("result").get("content")),
            "error": result.get("result").get("error"),
            "tool_call_id": call_id,
        }

        # Check for blocking guardrails
        guardrailing_result = check_blocking_guardrails(message, result)
        print("Guardrails check tool output result:", guardrailing_result)

        if len(guardrailing_result["errors"]) > 0:
            result["result"]["content"] = [
                {
                    "type": "text",
                    "text": "[Invariant] Your MCP tool call was blocked for security reasons. Do not attempt to circumvent this block, rather explain to the user based on the following output what went wrong: \n"
                    + json.dumps(guardrailing_result["errors"]),
                }
            ]

        append_and_push_trace(message)

        return result
    elif method == "tools/list":
        TOOLS = result.get("result").get("tools")
        return result
    else:
        return result


def forward_stdout(process, buffer_size=1):
    """Read from the process stdout, parse JSON chunks, and forward to sys.stdout"""
    buffer = b""
    decoder = json.JSONDecoder()

    while True:
        chunk = process.stdout.read(buffer_size)
        if not chunk:
            break
        buffer += chunk

        try:
            # Try parsing full JSON object from buffer
            text = buffer.decode("utf-8")
            obj = json.loads(text)

            obj = hook_tool_result(obj)
            # clear the buffer
            buffer = b""

            # Forward the original JSON to stdout
            json_output = json.dumps(obj).encode("utf-8") + b"\n"
            sys.stdout.buffer.write(json_output)
            sys.stdout.buffer.flush()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Wait for more data
            continue


def forward_stderr(process, buffer_size=1):
    """Read from the process stderr and write to sys.stderr"""
    for line in iter(lambda: process.stderr.read(buffer_size), b""):
        LOG_OUT.buffer.write(line)
        LOG_OUT.buffer.flush()


def fetch_guardrails(dataset):
    # Use async fetch_guardrails_from_explorer in a thread
    return asyncio.run(
        fetch_guardrails_from_explorer(
            dataset, "Bearer " + os.getenv("INVARIANT_API_KEY")
        )
    )


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mcp-implementation> [args...]")
        sys.exit(1)

    # Start the actual MCP implementation
    cmd = [sys.argv[1]] + sys.argv[2:]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # No buffering
    )

    # Ensure we have an INVARIANT_API_KEY
    if "INVARIANT_API_KEY" not in os.environ:
        print("INVARIANT_API_KEY environment variable is not set.")
        sys.exit(1)

    # Start threads to forward stdout and stderr
    stdout_thread = threading.Thread(
        target=forward_stdout, args=(process,), daemon=True
    )
    stderr_thread = threading.Thread(
        target=forward_stderr, args=(process,), daemon=True
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
                    id_to_method_mapping[obj.get("id")] = obj.get("method")

                # Check if this is a tool call request
                if obj.get("method") == "tools/call":
                    # Intercept and potentially modify the request
                    obj = hook_tool_call(obj)
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
        # Clean termination on Ctrl+C
        process.terminate()

    # Wait for process to terminate
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()

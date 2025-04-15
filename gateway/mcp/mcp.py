"""Gateway for MCP (Model Context Protocol) integration with Invariant."""

import sys
import subprocess
import json
import os
import threading

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.append_messages import AppendMessagesRequest
from invariant_sdk.types.push_traces import PushTracesRequest

from gateway.common.guardrails import GuardrailAction
from gateway.common.request_context import RequestContext
from gateway.integrations.explorer import create_annotations_from_guardrails_errors
from gateway.integrations.guardrails import check_guardrails
from gateway.mcp.log import mcp_log, MCP_LOG_FILE
from gateway.mcp.mcp_context import McpContext
from gateway.mcp.task_utils import run_task_in_background, run_task_sync

MCP_METHOD = "method"
UTF_8_ENCODING = "utf-8"
MCP_TOOL_CALL = "tools/call"
MCP_LIST_TOOLS = "tools/list"
INVARIANT_GUARDRAILS_BLOCKED_MESSAGE = """
                    [Invariant Guardrails] The MCP tool call was blocked for security reasons. 
                    Do not attempt to circumvent this block, rather explain to the user based 
                    on the following output what went wrong: %s
                    """


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


async def append_and_push_trace(
    ctx: McpContext, message: dict, guardrails_result: dict
) -> None:
    """
    Append a message to the trace if it exists or create a new one
    and push it to the Invariant Explorer.

    This function runs asynchronously in the background.
    """

    annotations = []
    if guardrails_result and guardrails_result.get("errors", []):
        annotations = create_annotations_from_guardrails_errors(
            guardrails_result["errors"]
        )

    if ctx.guardrails.logging_guardrails:
        logging_guardrails_check_result = get_guardrails_check_result(
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
        client = AsyncClient()
        if ctx.trace_id is None:
            ctx.trace.append(message)
            response = await client.push_trace(
                PushTracesRequest(
                    messages=[ctx.trace],
                    dataset=ctx.explorer_dataset,
                    metadata=[{"source": "mcp", "tools": ctx.tools}],
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


def get_guardrails_check_result(
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


def hook_tool_call(ctx: McpContext, request: dict) -> tuple[dict, bool]:
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

    # Check for blocking guardrails - this blocks until completion
    guardrailing_result = get_guardrails_check_result(
        ctx, message, action=GuardrailAction.BLOCK
    )

    # If the request is blocked, return a message indicating the block reason.
    # If there are new errors, run append_and_push_trace in background.
    # If there are no new errors, just return the original request.
    if (
        guardrailing_result
        and guardrailing_result.get("errors", [])
        and check_if_new_errors(ctx, guardrailing_result)
    ):
        if ctx.push_explorer:
            run_task_in_background(
                append_and_push_trace, ctx, message, guardrailing_result
            )
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32600,
                "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                % guardrailing_result["errors"],
            },
        }, True

    # Add the message to the trace
    ctx.trace.append(message)
    return request, False


def hook_tool_result(ctx: McpContext, result: dict) -> dict:
    """
    Hook function to intercept tool results.
    Modify this function to change behavior for tool results.
    Returns the potentially modified result.
    """
    method = ctx.id_to_method_mapping.get(result.get("id"))
    call_id = f"call_{result.get('id')}"

    if method is None:
        return result
    elif method == MCP_TOOL_CALL:
        message = {
            "role": "tool",
            "content": result.get("result").get("content"),
            "error": result.get("result").get("error"),
            "tool_call_id": call_id,
        }
        # Check for blocking guardrails - this blocks until completion
        guardrailing_result = get_guardrails_check_result(
            ctx, message, action=GuardrailAction.BLOCK
        )

        if guardrailing_result and guardrailing_result.get("errors", []):
            result = {
                "jsonrpc": "2.0",
                "id": result.get("id"),
                "error": {
                    "code": -32600,
                    "message": INVARIANT_GUARDRAILS_BLOCKED_MESSAGE
                    % guardrailing_result["errors"],
                },
            }

        if ctx.push_explorer:
            # Run append_and_push_trace in background
            run_task_in_background(
                append_and_push_trace, ctx, message, guardrailing_result
            )
        return result
    elif method == MCP_LIST_TOOLS:
        ctx.tools = result.get("result").get("tools")
        return result
    else:
        return result


def stream_and_forward_stdout(mcp_process: subprocess.Popen, ctx: McpContext) -> None:
    """Read from the mcp_process stdout, apply guardrails and and forward to sys.stdout"""
    for line in iter(mcp_process.stdout.readline, b""):
        try:
            # Process complete JSON lines
            line_str = line.decode(UTF_8_ENCODING).strip()
            if not line_str:
                continue

            parsed_json = json.loads(line_str)
            processed_json = hook_tool_result(ctx, parsed_json)

            # Write and flush immediately
            sys.stdout.buffer.write(write_as_utf8_bytes(processed_json))
            sys.stdout.buffer.flush()
        except json.JSONDecodeError as je:
            mcp_log(f"[ERROR] JSON decode error in stdout processing: {str(je)}")
            mcp_log(f"[ERROR] Problematic line: {line[:200]}...")

        except Exception as e:
            mcp_log(f"[ERROR] Error in stream_and_forward_stdout: {str(e)}")
            if line:
                mcp_log(f"[ERROR] Problematic line causing error: {line[:200]}...")


def stream_and_forward_stderr(
    mcp_process: subprocess.Popen, ctx: McpContext, read_chunk_size: int = 1
) -> None:
    """Read from the mcp_process stderr and write to sys.stderr"""
    for line in iter(lambda: mcp_process.stderr.read(read_chunk_size), b""):
        MCP_LOG_FILE.buffer.write(line)
        MCP_LOG_FILE.buffer.flush()


def run_stdio_input_loop(ctx: McpContext, mcp_process: subprocess.Popen) -> None:
    """Handle standard input, intercept call and forward requests to mcp_process stdin."""
    try:
        current_chunk = b""

        while True:
            buffer_input = sys.stdin.buffer.read(1)
            current_chunk += buffer_input

            if not buffer_input:
                break

            # Try to decode and parse as JSON to check for tool calls
            try:
                text = current_chunk.decode(UTF_8_ENCODING)
                parsed_json = json.loads(text)
                # clear the current chunk after successful parse
                current_chunk = b""
                # Refresh guardrails
                run_task_sync(ctx.load_guardrails)

                if parsed_json.get(MCP_METHOD) is not None:
                    ctx.id_to_method_mapping[parsed_json.get("id")] = parsed_json.get(
                        MCP_METHOD
                    )

                # Check if this is a tool call request
                if parsed_json.get(MCP_METHOD) == MCP_TOOL_CALL:
                    # Intercept and potentially block modify the request
                    hook_tool_call_result, is_blocked = hook_tool_call(ctx, parsed_json)
                    if not is_blocked:
                        # If blocked, hook_tool_call_result contains the original request.
                        # Forward the request to the MCP process.
                        # It will handle the request and return a response.
                        mcp_process.stdin.write(
                            write_as_utf8_bytes(hook_tool_call_result)
                        )
                        mcp_process.stdin.flush()
                    else:
                        # If blocked, hook_tool_call_result contains the block message.
                        # Forward the block message result back to the caller.
                        # The original request is not passed to the MCP process.
                        sys.stdout.buffer.write(
                            write_as_utf8_bytes(hook_tool_call_result)
                        )
                        sys.stdout.buffer.flush()
                    continue
                else:
                    mcp_process.stdin.write(write_as_utf8_bytes(parsed_json))
                    mcp_process.stdin.flush()
                    continue
            except Exception:
                # Not a complete or valid JSON, just pass through
                pass

    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        mcp_process.terminate()


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

    mcp_gateway_args, mcp_server_command_args = split_args(args)
    ctx = McpContext(mcp_gateway_args)

    mcp_process = subprocess.Popen(
        mcp_server_command_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Start threads to forward stdout and stderr
    threading.Thread(
        target=stream_and_forward_stdout,
        args=(mcp_process, ctx),
        daemon=True,
    ).start()
    threading.Thread(
        target=stream_and_forward_stderr,
        args=(mcp_process, ctx),
        daemon=True,
    ).start()

    # Handle forwarding stdin and intercept tool calls
    run_stdio_input_loop(ctx, mcp_process)

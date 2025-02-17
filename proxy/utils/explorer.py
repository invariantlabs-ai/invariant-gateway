"""Utility functions for the Invariant explorer."""

import os
from typing import Any, Dict, List
import re

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.push_traces import (
    PushTracesRequest,
    PushTracesResponse,
    AnnotationCreate,
)
import json

from invariant.analyzer import Policy

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


class PromptPatch(Exception):
    def __init__(self, patch=""):
        self.patch = patch


async def push_trace(
    messages: List[List[Dict[str, Any]]],
    dataset_name: str,
    invariant_authorization: str,
    dry: bool = False,
) -> PushTracesResponse:
    """Pushes traces to the dataset on the Invariant Explorer.

    If a dataset with the given name does not exist, it will be created.

    Args:
        messages (List[List[Dict[str, Any]]]): List of messages to push
        dataset_name (str): Name of the dataset.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        PushTracesResponse: Response containing the trace ID details.
    """
    # Remove any None values from the messages
    update_messages = [
        [{k: v for k, v in msg.items() if v is not None} for msg in msg_list]
        for msg_list in messages
    ]

    client = AsyncClient(
        api_url=os.getenv("INVARIANT_API_URL", DEFAULT_API_URL).rstrip("/"),
        api_key=invariant_authorization.split("Bearer ")[1],
    )

    # validate guardrails (and get annotations)
    blocked_and_annotations = [
        await validate_guardrails(client, messages, dataset_name)
        for messages in update_messages
    ]

    annotations = [annotation for _, annotation in blocked_and_annotations]
    blocked = [block for block, _ in blocked_and_annotations]

    # for blocked messages histories, apply "system" message with [blocked] content
    for i, block in enumerate(blocked):
        if block:
            update_messages[i].append(
                {
                    "content": f"[Agent execution blocked by guardrail: {error_label(block)}]",
                    "role": "system",
                }
            )

    request = PushTracesRequest(
        messages=update_messages, dataset=dataset_name, annotations=annotations
    )

    try:
        # if dry run, don't push the trace (but still validate guardrails)
        if dry and not any(blocked):
            result = {"dry_run": True}
        else:
            result = await client.push_trace(request)
    except Exception as e:
        print(f"Failed to push trace: {e}")
        result = {"error": str(e)}

    return blocked, result


async def validate_guardrails(
    client: AsyncClient,
    messages: List[List[Dict[str, Any]]],
    dataset_name: str,
) -> PushTracesResponse:
    """Fetches and validates the guardrails for the given dataset.

    Args:
        messages (List[List[Dict[str, Any]]]): List of messages to push.
        dataset_name (str): Name of the dataset.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        PushTracesResponse: Response containing the trace ID details.
    """
    try:
        metadata = await client.get_dataset_metadata(dataset_name=dataset_name)
    except Exception as e:
        print(f"Failed to get dataset metadata: {e}")
        return False, []
    guardrails = json.loads(metadata.get("guardrails", "[]"))

    blocked = False

    # preprocess messages
    trace = [{**msg} for msg in messages]

    # if content is missing in a msg, set it to empty string
    for msg in trace:
        if "content" not in msg:
            msg["content"] = ""

    # get first system prompt message
    system_prompt = next(
        (msg["content"] for msg in trace if msg["role"] == "system" and msg["content"]),
        None,
    )
    system_prompt_content = system_prompt if system_prompt else ""

    annotations = []

    for guardrail in guardrails:
        if not guardrail.get("enabled", False):
            print(f"Skipping guardrail {guardrail['name']} as it is disabled")
            continue

        policy = Policy.from_string(guardrail["policy"])
        results = policy.analyze(trace)

        for error in results.errors:
            label = error_label(error)
            # check for action=[block|warn|...]
            if "action=" in label:
                action = label.split("action=")[1].split()[0]
                if action == "block":
                    blocked = error
            elif "patch=" in label:
                print("label is", label)
                patch = label.split("patch=", 1)[1]
                if patch not in system_prompt_content:
                    raise PromptPatch(patch)

            ranges = [range for range in error.ranges]

            if len(ranges) > 1:
                prefixfree_ranges = []
                # remove prefixes
                for range in ranges:
                    if not any(
                        range.json_path.startswith(prefix.json_path)
                        for prefix in ranges
                        if range != prefix
                    ):
                        prefixfree_ranges.append(range)
                ranges = prefixfree_ranges

            for range in ranges:
                address = "messages." + range.json_path
                old_address = address
                try:
                    # if there is .start and .end, also include them as :start-end
                    if (
                        range.start is not None
                        and range.end is not None
                        and "-" not in address
                    ):
                        address += f":{range.start}-{range.end}"
                    # handle special cases
                    # 1. if the address is messages.[0-9]+, change it to messages.[0-9]+.content:0-<end of content field>
                    elif re.match(r"messages\.\d+$", address) and "-" not in address:
                        end = len(trace[int(address.split(".")[1])]["content"])
                        address += ".content:0-" + str(end)
                    # 2. if address is messages.4.tool_calls.0, change it to messages.4.tool_calls.0.function.name:0-<end of function name>
                    elif (
                        re.match(r"messages\.\d+\.tool_calls\.\d+$", address)
                        and "-" not in address
                    ):
                        end = len(
                            trace[int(address.split(".")[1])]["tool_calls"][
                                int(address.split(".")[3])
                            ]["function"]["name"]
                        )
                        address += ".function.name:0-" + str(end)
                except Exception as e:
                    print(f"Error while handling special cases: {e}")
                    pass

                print("fixed", old_address, "->", address)

                annotations.append(
                    AnnotationCreate(
                        content=label,
                        address=address,
                        extra_metadata={"source": "guardrail"},
                    )
                )
    return blocked, annotations


def error_label(error):
    if str(error).startswith("ErrorInformation("):
        return str(error).split("(", 1)[1].rsplit(")", 1)[0]
    return str(error)

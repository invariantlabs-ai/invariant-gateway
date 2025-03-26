"""Utility functions for the Invariant explorer."""

import os
from typing import Any, Dict, List

from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.push_traces import PushTracesRequest, PushTracesResponse
from invariant_sdk.types.annotations import AnnotationCreate

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


def create_annotations_from_guardrails_errors(
    guardrails_errors: List[dict],
) -> List[AnnotationCreate]:
    """Create Explorer annotations from the guardrails errors."""
    annotations = []

    def _remove_prefixes(ranges: list[str]) -> list[str]:
        """
        Remove prefixes from the list of ranges.

        If the ranges are ['messages.2', 'messages.2.content:25-30', 'messages.2.content']
        then this returns ['messages.2.content:25-30'].
        """
        ranges = sorted(ranges, key=len)
        result = []

        for i, s in enumerate(ranges):
            is_prefix = False
            for t in ranges[i + 1 :]:
                if t.startswith(s) and t != s:
                    is_prefix = True
                    break
            if not is_prefix:
                result.append(s)

        return result

    for error in guardrails_errors:
        content = error.get("args")[0]
        filtered_ranges = _remove_prefixes(list(error.get("ranges", [])))
        for r in filtered_ranges:
            annotations.append(
                AnnotationCreate(
                    content=content,
                    address=r,
                    extra_metadata={"source": "guardrails-error"},
                )
            )
    return annotations


async def push_trace(
    messages: List[List[Dict[str, Any]]],
    dataset_name: str,
    invariant_authorization: str,
    annotations: List[List[AnnotationCreate]] = None,
    metadata: List[Dict[str, Any]] = None,
) -> PushTracesResponse:
    """Pushes traces to the dataset on the Invariant Explorer.

    If a dataset with the given name does not exist, it will be created.

    Args:
        messages (List[List[Dict[str, Any]]]): List of messages to push.
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
    request = PushTracesRequest(
        messages=update_messages,
        annotations=annotations,
        dataset=dataset_name,
        metadata=metadata,
    )
    client = AsyncClient(
        api_url=os.getenv("INVARIANT_API_URL", DEFAULT_API_URL).rstrip("/"),
        api_key=invariant_authorization.split("Bearer ")[1],
    )
    try:
        return await client.push_trace(request)
    except Exception as e:
        print(f"Failed to push trace: {e}")
        return {"error": str(e)}

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
    for error in guardrails_errors:
        content = error.get("args")[0]
        for r in error.get("ranges", []):
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
        messages=update_messages, annotations=annotations, dataset=dataset_name
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

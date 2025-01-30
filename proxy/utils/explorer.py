"""Utility functions for the Invariant explorer."""

import os
from typing import Any, Dict, List

from fastapi import HTTPException
from invariant_sdk.client import Client

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"


def push_trace(
    messages: List[Dict[str, Any]],
    dataset_name: str,
    invariant_authorization: str,
) -> Dict[str, str]:
    """Pushes traces to the dataset on the Invariant Explorer.

    Args:
        messages (List[Dict[str, Any]]): List of messages to push.
        dataset_name (str): Name of the dataset.
        invariant_authorization (str): Authorization token.

    Returns:
        Dict[str, str]: Response containing the trace ID.
    """
    api_url = os.getenv("INVARIANT_API_URL", DEFAULT_API_URL)
    api_key = invariant_authorization.split("Bearer ")[1]
    client = Client(api_url=api_url, api_key=api_key)
    try:
        push_trace_response = client.create_request_and_push_trace(
            messages=messages, dataset=dataset_name
        )
        return {"trace_id": push_trace_response.id[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

"""Utility functions for the Invariant explorer."""

import os
from typing import Any, Dict, List

import httpx
from invariant_sdk.types.push_traces import PushTracesRequest

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"
PUSH_ENDPOINT = "/api/v1/push/trace"


async def push_trace(
    messages: List[Dict[str, Any]],
    dataset_name: str,
    invariant_authorization: str,
) -> Dict[str, str]:
    """Pushes traces to the dataset on the Invariant Explorer.

    Args:
        messages (List[Dict[str, Any]]): List of messages to push.
        dataset_name (str): Name of the dataset.
        invariant_authorization (str): Authorization token from the
                                       invariant-authorization header.

    Returns:
        Dict[str, str]: Response containing the trace ID.
    """
    api_url = os.getenv("INVARIANT_API_URL", DEFAULT_API_URL).rstrip("/")

    request = PushTracesRequest(messages=messages, dataset=dataset_name)
    async with httpx.AsyncClient() as client:
        explorer_push_request = client.build_request(
            "POST",
            f"{api_url}{PUSH_ENDPOINT}",
            json=request.to_json(),
            headers={
                "Authorization": f"{invariant_authorization}",
                "Accept": "application/json",
            },
        )
        try:
            response = await client.send(explorer_push_request)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"Failed to push trace: {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error pushing trace: {str(e)}")
            return {"error": str(e)}

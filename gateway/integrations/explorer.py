"""Utility functions for the Invariant explorer."""

import os
import json
from typing import Any

import httpx
from fastapi import HTTPException

from gateway.common.constants import DEFAULT_API_URL
from gateway.common.guardrails import GuardrailRuleSet, Guardrail, GuardrailAction
from invariant_sdk.async_client import AsyncClient
from invariant_sdk.types.push_traces import PushTracesRequest, PushTracesResponse
from invariant_sdk.types.annotations import AnnotationCreate


def create_annotations_from_guardrails_errors(
    guardrails_errors: list[dict],
) -> list[AnnotationCreate]:
    """Create Explorer annotations from the guardrails errors."""
    annotations = []

    def _pick_most_specific_ranges(ranges: list[str]) -> list[str]:
        """
        Remove redundant prefixes from the list of ranges.

        If the ranges are ['messages.2', 'messages.2.content:25-30', 'messages.2.content']
        then this returns ['messages.2.content:25-30'].

        This picks the most specific subset of the ranges and removes the rest. If some
        range is a proper prefix of another range, it is removed.
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
        filtered_ranges = _pick_most_specific_ranges(list(error.get("ranges", [])))
        for r in filtered_ranges:
            annotations.append(
                AnnotationCreate(
                    content=content,
                    address=r,
                    extra_metadata={
                        "source": "guardrails-error",
                        # if included in error, also include information about guardrail source
                        **(
                            {"guardrail": error.get("guardrail")}
                            if error.get("guardrail")
                            else {}
                        ),
                    },
                )
            )
    # Remove duplicates
    return remove_duplicates(annotations)


def remove_duplicates(annotations: list[AnnotationCreate]) -> list[AnnotationCreate]:
    """
    Remove duplicate annotations based on content, address, and extra_metadata.

    Two annotations are considered duplicates if they have the same content,
    address, and extra_metadata.
    """
    unique_annotations = []
    seen = set()

    for annotation in annotations:
        # Convert the entire extra_metadata dict to a JSON string
        # This creates a hashable representation regardless of nested content
        metadata_str = json.dumps(annotation.extra_metadata or {}, sort_keys=True)

        # Create a unique identifier using all three fields
        unique_key = (annotation.content, annotation.address, metadata_str)

        if unique_key not in seen:
            seen.add(unique_key)
            unique_annotations.append(annotation)

    return unique_annotations


def get_explorer_api_url() -> str:
    """Get the Invariant Explorer API URL from the environment variable."""
    return os.getenv("INVARIANT_API_URL", DEFAULT_API_URL)


async def push_trace(
    messages: list[list[dict[str, Any]]],
    dataset_name: str,
    invariant_authorization: str,
    annotations: list[list[AnnotationCreate]] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> PushTracesResponse:
    """Pushes traces to the dataset on the Invariant Explorer.

    If a dataset with the given name does not exist, it will be created.

    Args:
        messages (listlistdict[str, Any]]]): List of messages to push.
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
        api_url=get_explorer_api_url().rstrip("/"),
        api_key=invariant_authorization.split("Bearer ")[1],
    )
    try:
        return await client.push_trace(request)
    except Exception as e: # pylint: disable=broad-except
        print(f"Failed to push trace: {e}")
        return {"error": str(e)}


async def fetch_guardrails_from_explorer(
    dataset_name: str,
    invariant_authorization: str,
    client_name: str | None = None,
    server_name: str | None = None,
) -> GuardrailRuleSet:
    """Get the guardrails for the dataset.

    Returns:
        GuardrailRuleSet: The guardrails for the dataset grouped by their action.
    """

    # TODO: Implement a single API in explorer backend which can return
    # dataset details without requiring a username.

    client = httpx.AsyncClient(
        base_url=get_explorer_api_url().rstrip("/"),
        headers={
            "Authorization": invariant_authorization,
        },
    )

    # Get the user details.
    user_info_response = await client.get("/api/v1/user/identity", timeout=5)
    if user_info_response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Invalid Invariant API key. Please check your API key.",
        )
    elif user_info_response.status_code != 200:
        raise ValueError(
            f"Failed to get user details from Explorer: {user_info_response.status_code}, {user_info_response.text}"
        )
    user_details = user_info_response.json()
    username = user_details["username"]

    # Get the dataset policies.
    policies_response = await client.get(
        f"/api/v1/dataset/byuser/{username}/{dataset_name}/policy",
        params={
            **({"client_name": client_name} if client_name else {}),
            **({"server_name": server_name} if server_name else {}),
        },
    )
    if policies_response.status_code != 200:
        if policies_response.status_code == 404:
            # If the dataset does not exist, return empty guardrails.
            return GuardrailRuleSet(
                blocking_guardrails=[],
                logging_guardrails=[],
            )
        raise ValueError(
            f"Failed to get dataset details from Explorer: {policies_response.status_code}, {policies_response.text}"
        )
    policies_details = policies_response.json()
    guardrails = policies_details.get("policies", [])

    blocking_guardrails = []
    logging_guardrails = []
    for g in guardrails:
        action = g["action"]

        if not g["enabled"]:
            # Skip guardrails that are not enabled.
            continue

        if action not in (GuardrailAction.BLOCK, GuardrailAction.LOG):
            print("[Warning] Skipping unknown guardrail action: ", action)
            continue

        guardrail = Guardrail(
            id=g["id"],
            name=g["name"],
            content=g["content"],
            action=GuardrailAction(action),
        )

        if action == GuardrailAction.BLOCK:
            blocking_guardrails.append(guardrail)
        else:
            logging_guardrails.append(guardrail)

    return GuardrailRuleSet(
        blocking_guardrails=blocking_guardrails,
        logging_guardrails=logging_guardrails,
    )

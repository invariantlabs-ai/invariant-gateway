"""Utility functions for Guardrails execution."""

import os
from typing import Any, Dict, List

import httpx

DEFAULT_API_URL = "https://guardrail.invariantnet.com"


async def check_guardrails(
    messages: List[Dict[str, Any]], guardrails: str, invariant_authorization: str
) -> Dict[str, Any]:
    """
    Checks guardrails on the list of messages.

    Args:
        messages (List[Dict[str, Any]]): List of messages to verify the guardrails against.
        guardrails (str): The guardrails to check against.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        Dict: Response containing guardrail check results.
    """
    client = httpx.AsyncClient()
    url = os.getenv("GUADRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
    try:
        result = await client.post(
            f"{url}/api/v1/policy/check",
            json={"messages": messages, "policy": guardrails},
            headers={
                "Authorization": invariant_authorization,
                "Accept": "application/json",
            },
        )
        print(f"Guardrail check response: {result.json()}")
        return result.json()
    except Exception as e:
        print(f"Failed to verify guardrails: {e}")
        return {"error": str(e)}

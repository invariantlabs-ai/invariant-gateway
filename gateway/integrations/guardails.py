"""Utility functions for Guardrails execution."""

import asyncio
import os
import time
from typing import Any, Dict, List
from functools import wraps

import httpx

DEFAULT_API_URL = "https://guardrail.invariantnet.com"


# Timestamps of last API calls per guardrails string
_guardrails_cache = {}
# Locks per guardrails string
_guardrails_locks = {}


def rate_limit(expiration_time: int = 3600):
    """
    Decorator to limit API calls to once per expiration_time seconds
    per unique guardrails string.

    Args:
        expiration_time (int): Time in seconds to cache the guardrails.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(guardrails: str, *args, **kwargs):
            now = time.time()

            # Get or create a per-guardrail lock
            if guardrails not in _guardrails_locks:
                _guardrails_locks[guardrails] = asyncio.Lock()
            guardrail_lock = _guardrails_locks[guardrails]

            async with guardrail_lock:
                last_called = _guardrails_cache.get(guardrails)

                if last_called and (now - last_called < expiration_time):
                    # Skipping API call: Guardrails '{guardrails}' already
                    # preloaded within expiration_time
                    return

                # Update cache timestamp
                _guardrails_cache[guardrails] = now

            try:
                await func(guardrails, *args, **kwargs)
            finally:
                _guardrails_locks.pop(guardrails, None)

        return wrapper

    return decorator


@rate_limit(3600)  # Don't preload the same guardrails string more than once per hour
async def _preload(guardrails: str, invariant_authorization: str) -> None:
    """
    Calls the Guardrails API to preload the provided policy for faster checking later.

    Args:
        guardrails (str): The guardrails to preload.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.
    """
    async with httpx.AsyncClient() as client:
        url = os.getenv("GUADRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        result = await client.post(
            f"{url}/api/v1/policy/load",
            json={"policy": guardrails},
            headers={
                "Authorization": invariant_authorization,
                "Accept": "application/json",
            },
        )
        result.raise_for_status()


async def preload_guardrails(context: "RequestContextData") -> None:
    """
    Preloads the guardrails for faster checking later.

    Args:
        context: RequestContextData object.
    """
    if not context.config or not context.config.guardrails:
        return

    try:
        task = asyncio.create_task(
            _preload(context.config.guardrails, context.invariant_authorization)
        )
        asyncio.shield(task)
    except Exception as e:
        print(f"Error scheduling preload_guardrails task: {e}")


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
    async with httpx.AsyncClient() as client:
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

"""Utility functions for Guardrails execution."""

import asyncio
import os
import time
from functools import wraps
from typing import Any

import httpx
from fastapi import HTTPException

from gateway.common.constants import CONTENT_TYPE_JSON, DEFAULT_API_URL
from gateway.common.request_context import RequestContext
from gateway.common.authorization import (
    INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER,
)
from gateway.common.guardrails import Guardrail

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
        url = os.getenv("GUARDRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        result = await client.post(
            f"{url}/api/v1/policy/load",
            json={"policy": guardrails},
            headers={
                "Authorization": invariant_authorization,
                "Accept": CONTENT_TYPE_JSON,
            },
        )
        result.raise_for_status()


async def preload_guardrails(context: "RequestContext") -> None:
    """
    Preloads the guardrails for faster checking later.

    Args:
        context: RequestContext object.
    """
    if not context.guardrails:
        return

    try:
        # Move these calls to a batch preload/validate API.
        for blocking_guardrail in context.guardrails.blocking_guardrails:
            task = asyncio.create_task(
                _preload(
                    blocking_guardrail.content, context.get_guardrailing_authorization()
                )
            )
            asyncio.shield(task)
        for logging_guadrail in context.guardrails.logging_guardrails:
            task = asyncio.create_task(
                _preload(
                    logging_guadrail.content,
                    context.get_guardrailing_authorization(),
                )
            )
            asyncio.shield(task)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error scheduling preload_guardrails task: {e}")


async def check_guardrails(
    messages: list[dict[str, Any]],
    guardrails: list[Guardrail],
    context: RequestContext,
) -> dict[str, Any]:
    """
    Checks guardrails on the list of messages.
    This calls the batch check API of the Guardrails service.

    Args:
        messages (list[dict[str, Any]]): List of messages to verify the guardrails against.
        guardrails (list[Guardrail]): The guardrails to check against.
        invariant_authorization (str): Value of the
                                       invariant-authorization header.

    Returns:
        dict: Response containing guardrail check results.
    """
    async with httpx.AsyncClient() as client:
        url = os.getenv("GUARDRAILS_API_URL", DEFAULT_API_URL).rstrip("/")
        try:
            result = await client.post(
                f"{url}/api/v1/policy/check/batch",
                json={
                    "messages": messages,
                    "policies": [g.content for g in guardrails],
                    "parameters": context.guardrails_parameters or {},
                },
                headers={
                    "Authorization": context.get_guardrailing_authorization(),
                    "Accept": CONTENT_TYPE_JSON,
                },
                timeout=5,
            )
            if not result.is_success:
                if result.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail=(
                            "The provided Invariant API key is not valid for guardrail checking. "
                            "Please ensure you are using the correct API key or pass an "
                            "alternative API key for guardrail checking specifically via the "
                            f"'{INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER}' header."
                        ),
                    )
                raise Exception(  # pylint: disable=broad-exception-raised
                    f"Guardrails check failed: {result.status_code} - {result.text}"
                )
            guardrails_result = result.json()

            aggregated_errors = {"errors": []}
            for res, guardrail in zip(guardrails_result.get("result", []), guardrails):
                for error in res.get("errors", []):
                    # add each error to the aggregated errors but keep track
                    # of which guardrail it belongs to
                    aggregated_errors["errors"].append(
                        {
                            **error,
                            "guardrail": {
                                "id": guardrail.id,
                                "name": guardrail.name,
                                "content": guardrail.content,
                                "action": guardrail.action,
                            },
                        }
                    )

                # check for any error_message
                if error_message := res.get("error_message"):
                    return {
                        "errors": [
                            {"args": [error_message], "kwargs": {}, "ranges": []}
                        ]
                    }
            return aggregated_errors
        except HTTPException as e:
            raise e
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to verify guardrails: {e}")
            # make sure runtime errors are also visible in e.g. Explorer
            return {
                "errors": [
                    {
                        "args": ["Gateway: " + str(e)],
                        "kwargs": {},
                        "ranges": ["messages[0].content:L0"],
                    }
                ]
            }

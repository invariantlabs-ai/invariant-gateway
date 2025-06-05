"""Common Configurations for the Gateway Server."""

import asyncio
import os
import threading

import fastapi
from httpx import HTTPStatusError

from gateway.common.guardrails import Guardrail, GuardrailAction, GuardrailRuleSet


def extract_policy_from_headers(request: fastapi.Request | None) -> str | None:
    """
    Extracts the guardrailing policy from the request headers if present.

    Returns 'None' if no such header is present.
    """
    if request is None:
        return None

    policy = request.headers.get("Invariant-Guardrails")
    # undo unicode_escape
    if policy:
        # interpret as bytes then decode
        policy = policy.encode("utf-8").decode("unicode_escape")
    return policy


class GatewayConfig:
    """Common configurations for the Gateway Server."""

    def __init__(self):
        self.guardrails_from_file = self._load_guardrails_from_file()

    def _load_guardrails_from_file(self) -> str:
        """
        Loads the guardrails from the file specified in GUARDRAILS_FILE_PATH.
        Returns the guardrails file content as a string.
        """
        from gateway.integrations.guardrails import _preload

        guardrails_file = os.getenv("GUARDRAILS_FILE_PATH", "")
        if not guardrails_file:
            print("[warning: GUARDRAILS_FILE_PATH is not set. Using empty guardrails]")
            return ""

        invariant_api_key = os.getenv("INVARIANT_API_KEY", "")
        if not invariant_api_key:
            raise ValueError(
                "Error: INVARIANT_API_KEY is not set."
                "It is required to validate guardrails file content."
            )

        try:
            with open(guardrails_file, "r", encoding="utf-8") as f:
                guardrails_file_content = f.read()
                asyncio.run(
                    _preload(guardrails_file_content, "Bearer " + invariant_api_key)
                )
                return guardrails_file_content

        except (FileNotFoundError, PermissionError, OSError) as e:
            raise ValueError(
                f"Unable to read guardrails file ({guardrails_file}): {e}"
            ) from e
        except HTTPStatusError as e:
            raise ValueError(f"Cannot load guardrails, {e}, {e.response.text}") from e

    def __repr__(self) -> str:
        return f"GatewayConfig(guardrails_from_file={repr(self.guardrails_from_file)})"


class GatewayConfigManager:
    """Manager for Gateway Configuration."""

    _config_instance = None
    _lock = threading.Lock()

    @classmethod
    def get_config(cls) -> GatewayConfig:
        """Initializes and returns the gateway configuration using double-checked locking."""
        local_config = cls._config_instance

        if local_config is None:
            with cls._lock:
                local_config = cls._config_instance
                if local_config is None:
                    local_config = GatewayConfig()
                    cls._config_instance = local_config

        return local_config


async def extract_guardrails_from_header(
    request: fastapi.Request,
) -> GuardrailRuleSet | None:
    """
    Extracts Invariant-Guardrails from the request header if provided, and returns a corresponding
    GuardrailRuleSet. If no guardrails are provided, returns None.
    """
    # if provided in header, use custom guardrailing policy
    if guardrails := extract_policy_from_headers(request):
        guardrails = [
            Guardrail(
                id="guardrails-from-header",
                name="guardrails from request header",
                content=guardrails,
                action=GuardrailAction.BLOCK,
            )
        ]

        return GuardrailRuleSet(
            blocking_guardrails=guardrails,
            logging_guardrails=[],
        )
    return None

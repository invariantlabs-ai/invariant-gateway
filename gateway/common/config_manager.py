"""Common Configurations for the Gateway Server."""

import asyncio
import os
import threading

from integrations.guardails import _preload

from httpx import HTTPStatusError


class GatewayConfig:
    """Common configurations for the Gateway Server."""

    def __init__(self):
        self.guardrails = self._load_guardrails_from_file()

    def _load_guardrails_from_file(self) -> str:
        """
        Loads the guardrails from the file specified in GUARDRAILS_FILE_PATH.
        Returns the guardrails file content as a string.
        """
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
        return f"GatewayConfig(guardrails={repr(self.guardrails)})"


class GatewayConfigManager:
    """Manager for Gateway Configuration."""

    _config_instance = None
    _lock = threading.Lock()

    @classmethod
    def get_config(cls):
        """Initializes and returns the gateway configuration using double-checked locking."""
        local_config = cls._config_instance

        if local_config is None:
            with cls._lock:
                local_config = cls._config_instance
                if local_config is None:
                    local_config = GatewayConfig()
                    cls._config_instance = local_config
        return local_config

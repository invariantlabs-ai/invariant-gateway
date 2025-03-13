"""Common Configurations for the Gateway Server."""

import os
import threading

from invariant.analyzer import Policy


class GatewayConfig:
    """Common configurations for the Gateway Server."""

    def __init__(self):
        self.guardrails = self._load_guardrails()

    def _load_guardrails(self) -> str:
        """
        Loads and validates guardrails from the file specified in GUARDRAILS_FILE_PATH.
        Returns the guardrails file content as a string if valid; otherwise, raises an error.
        """
        guardrails_file = os.getenv("GUARDRAILS_FILE_PATH", "")

        if not guardrails_file:
            print("[warning: GUARDRAILS_FILE_PATH is not set. Using empty guardrails]")
            return ""

        try:
            with open(guardrails_file, "r", encoding="utf-8") as f:
                guardrails_file_content = f.read()
            _ = Policy.from_string(guardrails_file_content)
            return guardrails_file_content

        except (FileNotFoundError, PermissionError, OSError) as e:
            raise ValueError(
                f"Error: Unable to read guardrails file ({guardrails_file}): {e}"
            ) from e

        except Exception as e:
            raise ValueError(f"Invalid policy content in {guardrails_file}: {e}") from e

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

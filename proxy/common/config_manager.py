"""Common Configurations for the Proxy Server."""

import os
import threading

from invariant.analyzer import Policy


class ProxyConfig:
    """Common configurations for the Proxy Server."""

    def __init__(self):
        self.policies = self._load_policies()

    def _load_policies(self) -> str:
        """
        Loads and validates policies from the file specified in POLICIES_FILE_PATH.
        Returns the policy file content as a string if valid; otherwise, raises an error.
        """
        policies_file = os.getenv("POLICIES_FILE_PATH", "")

        if not policies_file:
            print("Warning: POLICIES_FILE_PATH is not set. Using empty policies.")
            return ""

        try:
            with open(policies_file, "r", encoding="utf-8") as f:
                policy_file_content = f.read()
            _ = Policy.from_string(policy_file_content)
            return policy_file_content

        except (FileNotFoundError, PermissionError, OSError) as e:
            raise ValueError(
                f"Error: Unable to read policies file ({policies_file}): {e}"
            ) from e

        except Exception as e:
            raise ValueError(f"Invalid policy content in {policies_file}: {e}") from e

    def __repr__(self) -> str:
        return f"ProxyConfig(policies={repr(self.policies)})"


class ProxyConfigManager:
    """Manager for Proxy Configuration."""

    _config_instance = None
    _lock = threading.Lock()

    @classmethod
    def get_config(cls):
        """Initializes and returns the proxy configuration using double-checked locking."""
        local_config = cls._config_instance

        if local_config is None:
            with cls._lock:
                local_config = cls._config_instance
                if local_config is None:
                    local_config = ProxyConfig()
                    cls._config_instance = local_config
        return local_config

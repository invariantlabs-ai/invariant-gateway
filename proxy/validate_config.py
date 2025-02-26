"""Validates the ProxyConfigManager configuration."""

import sys

from common.config_manager import ProxyConfigManager

try:
    _ = ProxyConfigManager.get_config()
    print("ProxyConfig validated successfully.")
    sys.exit(0)
except Exception as e:
    print(f"Error loading ProxyConfig error: {e}")
    sys.exit(1)

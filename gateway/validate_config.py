"""Validates the GatewayConfigManager configuration."""

import sys

from gateway.common.config_manager import GatewayConfigManager

try:
    _ = GatewayConfigManager.get_config()
    print("[gateway config validated successfully]")
    sys.exit(0)
except Exception as e:
    print(f"Error loading GatewayConfig: {e}")
    sys.exit(1)

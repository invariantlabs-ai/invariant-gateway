"""Common Request context data class."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config_manager import GatewayConfig


@dataclass(frozen=True)
class RequestContextData:
    """Request context data class."""

    request_json: Dict[str, Any]
    dataset_name: Optional[str] = None
    invariant_authorization: Optional[str] = None
    config: Optional[GatewayConfig] = None

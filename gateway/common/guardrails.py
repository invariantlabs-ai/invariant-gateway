"""Common guardrails data class."""

from dataclasses import dataclass
from enum import Enum

class GuardrailAction(str, Enum):
    """Enum representing the action to be taken for guardrail rules."""

    BLOCK = "block"
    LOG = "log"


@dataclass(frozen=True)
class Guardrail:
    """Represents a single guardrail rule."""

    id: str
    name: str
    content: str
    action: GuardrailAction


@dataclass(frozen=True)
class GuardrailRuleSet:
    """Grouped guardrail rules separated by their action."""

    blocking_guardrails: list[Guardrail]
    logging_guardrails: list[Guardrail]

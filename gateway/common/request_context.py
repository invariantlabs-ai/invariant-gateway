"""Common Request context data class."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from common.config_manager import GatewayConfig
from common.guardrails import GuardrailRuleSet, Guardrail, GuardrailAction


@dataclass(frozen=True)
class RequestContext:
    """Structured context for a request. Must be created via `RequestContext.create()`."""

    request_json: Dict[str, Any]
    dataset_name: Optional[str] = None
    invariant_authorization: Optional[str] = None
    # the set of guardrails to enforce for this request
    guardrails: Optional[GuardrailRuleSet] = None
    config: Dict[str, Any] = None

    _created_via_factory: bool = field(
        default=False, init=True, repr=False, compare=False
    )

    def __post_init__(self):
        if not self._created_via_factory:
            raise RuntimeError(
                "RequestContext must be created using RequestContext.create()"
            )

    @classmethod
    def create(
        cls,
        request_json: Dict[str, Any],
        dataset_name: Optional[str] = None,
        invariant_authorization: Optional[str] = None,
        guardrails: Optional[GuardrailRuleSet] = None,
        config: Optional[GatewayConfig] = None,
    ) -> "RequestContext":
        """Creates a new RequestContext instance, applying default guardrails if needed."""

        # Convert GatewayConfig to a basic dict, excluding guardrails_from_file
        context_config = {
            key: value
            for key, value in (config.__dict__.items() if config else {})
            if key != "guardrails_from_file"
        }

        # If no guardrails are configured for the dataset on Explorer,
        # and the config specifies guardrails_from_file, use that.
        guardrails = guardrails
        if (
            (
                not guardrails
                or (
                    not guardrails.blocking_guardrails
                    and not guardrails.logging_guardrails
                )
            )
            and config
            and config.guardrails_from_file
        ):
            # TODO: Support logging guardrails via file.
            guardrails = GuardrailRuleSet(
                blocking_guardrails=[
                    Guardrail(
                        id="default",
                        name="default",
                        content=config.guardrails_from_file,
                        action=GuardrailAction.BLOCK,
                    )
                ],
                logging_guardrails=[],
            )

        return cls(
            request_json=request_json,
            dataset_name=dataset_name,
            invariant_authorization=invariant_authorization,
            guardrails=guardrails,
            config=context_config,
            _created_via_factory=True,
        )

    def __repr__(self) -> str:
        return (
            f"RequestContext("
            f"request_json={self.request_json}, "
            f"dataset_name={self.dataset_name}, "
            f"invariant_authorization={self.invariant_authorization}, "
            f"guardrails={self.guardrails}, "
            f"config={self.config})"
        )

"""Common Request context data class."""

from dataclasses import dataclass, field
from typing import Any

import fastapi

from gateway.common.authorization import (
    extract_guardrail_service_authorization_from_headers,
)
from gateway.common.config_manager import GatewayConfig
from gateway.common.guardrails import GuardrailRuleSet, Guardrail, GuardrailAction


@dataclass(frozen=True)
class RequestContext:
    """Structured context for a request. Must be created via `RequestContext.create()`."""

    request_json: dict[str, Any]
    dataset_name: str | None = None
    # authorization to use for invariant service like explorer
    invariant_authorization: str | None = None
    # authorization to use for invariant guardrailing specifically
    guardrail_authorization: str | None = None
    # the set of guardrails to enforce for this request
    guardrails: GuardrailRuleSet | None = None
    config: dict[str, Any] | None = None

    # extra parameters available as input.<key> during guardrail evaluation
    guardrails_parameters: dict[str, Any] | None = None

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
        request_json: dict[str, Any],
        dataset_name: str | None = None,
        invariant_authorization: str | None = None,
        guardrails: GuardrailRuleSet | None = None,
        config: GatewayConfig | None = None,
        request: fastapi.Request | None = None,
        guardrails_parameters: dict[str, Any] | None = None,
    ) -> "RequestContext":
        """Creates a new RequestContext instance, applying default guardrails if needed."""

        # Convert GatewayConfig to a basic dict, excluding guardrails_from_file
        context_config = {
            key: value
            for key, value in (config.__dict__.items() if config else {})
            if key != "guardrails_from_file"
        }

        # If no guardrails are configured and the config specifies
        # guardrails_from_file, use those instead.
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
            guardrails = GuardrailRuleSet(
                blocking_guardrails=[
                    Guardrail(
                        id="guardrails-from-gateway-config-file",
                        name="guardrails from gateway configuration file",
                        content=config.guardrails_from_file,
                        action=GuardrailAction.BLOCK,
                    )
                ],
                logging_guardrails=[],
            )

        # if additionally provided, extract separate API key to use with guardrailing service
        guardrail_service_authorization = None
        if request and (
            guardrail_authorization
            := extract_guardrail_service_authorization_from_headers(request)
        ):
            guardrail_service_authorization = guardrail_authorization

        return cls(
            request_json=request_json,
            dataset_name=dataset_name,
            invariant_authorization=invariant_authorization,
            guardrail_authorization=guardrail_service_authorization,
            guardrails=guardrails,
            config=context_config,
            _created_via_factory=True,
            guardrails_parameters=guardrails_parameters,
        )

    def get_guardrailing_authorization(self) -> str | None:
        """
        Returns the authorization to use for the guardrailing service.

        This can be different from the invariant authorization, but falls back
        "to be the same if not explicitly set via header.

        See also extract_guardrail_service_authorization_from_headers(...)
        """

        return self.guardrail_authorization or self.invariant_authorization

    def __repr__(self) -> str:
        return (
            f"RequestContext("
            f"request_json={self.request_json}, "
            f"dataset_name={self.dataset_name}, "
            f"invariant_authorization=inv-*****{self.invariant_authorization[-4:]}, "
            f"guardrails={self.guardrails}, "
            f"config={self.config})"
        )

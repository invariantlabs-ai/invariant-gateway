"""Common Authorization functions used in the gateway."""

from fastapi import HTTPException, Request

INVARIANT_AUTHORIZATION_HEADER = "invariant-authorization"
INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER = "invariant-guardrails-authorization"
API_KEYS_SEPARATOR = ";invariant-auth="


def extract_guardrail_service_authorization_from_headers(
    request: Request,
) -> tuple[str | None, str | None]:
    """
    Extracts the optional Invariant-Guardrails-Authorization authorization header from the request.

    This header can be specifified to use a different API key for guardrailing compared to
    Explorer interactions.
    """
    return request.headers.get(INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER)


def extract_authorization_from_headers(
    request: Request,
    dataset_name: str | None = None,
    llm_provider_api_key_header: str | None = None,
    llm_provider_fallback_api_key_headers: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Extracts the Invariant authorization and LLM Provider API key from the request headers.

    In case the user wants to push to Explorer (when dataset_name is not None),
    the request headers must contain the Invariant API Key.
    The invariant-authorization header contains the Invariant API Key as
    "invariant-authorization": "Bearer <Invariant API Key>"
    {llm_provider_api_key_header} contains the LLM Provider API Key as
    {llm_provider_api_key_header}: "<API Key>"

    If {llm_provider_api_key_header} is not among headers, we look for
    any header among {llm_provider_fallback_api_key_headers}.

    For some clients, it is not possible to pass a custom header at all,
    In such cases, the Invariant API Key is passed as part of the
    {llm_provider_api_key_header} with the LLM Provider API Key
    The header in that case becomes:
    {llm_provider_api_key_header}: "<API Key>;invariant-auth=<Invariant API Key>"
    """
    # invariant api key
    invariant_authorization = request.headers.get(INVARIANT_AUTHORIZATION_HEADER)

    # llm provider api key (also check fallbacks for clients like litellm)
    if llm_provider_api_key_header is not None:
        llm_provider_api_key = request.headers.get(llm_provider_api_key_header)

        if llm_provider_api_key is None and llm_provider_fallback_api_key_headers:
            for header in llm_provider_fallback_api_key_headers:
                llm_provider_api_key = request.headers.get(header)
                if llm_provider_api_key:
                    llm_provider_api_key_header = header
                    break
    else:
        llm_provider_api_key = None

    # if the dataset name is not None, we need to check if the invariant api key is present
    if dataset_name:
        if invariant_authorization is None:
            if llm_provider_api_key is None:
                raise HTTPException(
                    status_code=400, detail="Missing LLM Provider API Key"
                )

            if API_KEYS_SEPARATOR not in llm_provider_api_key:
                raise HTTPException(status_code=400, detail="Missing invariant api key")

            # Both the API keys are passed in the llm_provider_api_key_header
            api_keys = request.headers.get(llm_provider_api_key_header).split(
                API_KEYS_SEPARATOR
            )
            if len(api_keys) != 2 or not api_keys[1].strip():
                raise HTTPException(status_code=400, detail="Invalid API Key format")

            invariant_authorization = f"Bearer {api_keys[1].strip()}"
            llm_provider_api_key = f"{api_keys[0].strip()}"

    if llm_provider_api_key and "Bearer " in llm_provider_api_key:
        llm_provider_api_key = llm_provider_api_key.split("Bearer ")[1].strip()

    return invariant_authorization, llm_provider_api_key

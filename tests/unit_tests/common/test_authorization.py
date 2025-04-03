"""Tests for the authorization header extractor."""

import os
import sys
from fastapi import HTTPException
import random
import string
import pytest

from gateway.common.config_manager import GatewayConfig
from gateway.common.request_context import RequestContext


# Add root folder (parent) to sys.path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from gateway.common.authorization import (
    INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER,
    extract_authorization_from_headers,
    INVARIANT_AUTHORIZATION_HEADER,
    API_KEYS_SEPARATOR,
)


@pytest.mark.parametrize("push_to_explorer", [True, False])
@pytest.mark.parametrize("invariant_authorization", [True, False])
@pytest.mark.parametrize(
    "invariant_authorization_appended_to_llm_provider_api_key", [True, False]
)
@pytest.mark.parametrize("use_fallback_header", [True, False])
def test_extract_authorization_from_headers(
    push_to_explorer: bool,
    invariant_authorization: bool,
    invariant_authorization_appended_to_llm_provider_api_key: bool,
    use_fallback_header: bool,
):
    """Test the extract_authorization_from_headers function."""

    llm_apikey = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    inv_apikey = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    dataset_name = "test-dataset" if push_to_explorer else None

    headers: dict[str, str] = {}

    llm_provider_api_key = (
        "fallback-header" if use_fallback_header else "llm-provider-api-key"
    )
    headers[llm_provider_api_key] = llm_apikey

    if invariant_authorization:
        print("invariant_authorization - TRUE")
        if invariant_authorization_appended_to_llm_provider_api_key:
            headers[llm_provider_api_key] = (
                f"{headers.get(llm_provider_api_key, '')}{API_KEYS_SEPARATOR}{inv_apikey}"
            )
        else:
            headers[INVARIANT_AUTHORIZATION_HEADER] = f"Bearer {inv_apikey}"

    # Mock request headers
    class MockRequest:
        def __init__(self, headers):
            self.headers = headers

    request = MockRequest(headers)

    # Call the function
    try:
        invariant_auth, llm_provider_api_key = extract_authorization_from_headers(
            request,
            dataset_name=dataset_name,
            llm_provider_api_key_header="llm-provider-api-key",
            llm_provider_fallback_api_key_headers=["fallback-header"],
        )
    except HTTPException as e:
        # If an exception is raised, check if it is the expected one
        if not invariant_authorization:
            assert e.status_code == 400
            assert e.detail == "Missing invariant api key"
            return
        else:
            raise e
    # Verify the results
    if invariant_authorization:
        if (
            not push_to_explorer
            and invariant_authorization_appended_to_llm_provider_api_key
        ):
            assert llm_provider_api_key.split(API_KEYS_SEPARATOR)[0] == llm_apikey
            assert llm_provider_api_key.split(API_KEYS_SEPARATOR)[1] == inv_apikey
        else:
            assert invariant_auth == ("Bearer " + inv_apikey)
    else:
        assert invariant_auth is None
    if not (
        not push_to_explorer
        and invariant_authorization_appended_to_llm_provider_api_key
    ):
        assert llm_provider_api_key == llm_apikey


@pytest.mark.parametrize("use_guardrailing_api_key", [True, False])
def test_extract_guardrails_authorization_from_headers(use_guardrailing_api_key: bool):
    headers: dict[str, str] = {}

    inv_apikey = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    inv_guardrails_apikey = "".join(
        random.choices(string.ascii_letters + string.digits, k=10)
    )
    llm_apikey = "".join(random.choices(string.ascii_letters + string.digits, k=10))

    headers[INVARIANT_AUTHORIZATION_HEADER] = f"Bearer {inv_apikey}"
    headers["Authorization"] = f"Bearer {llm_apikey}"

    if use_guardrailing_api_key:
        headers[INVARIANT_GUARDRAIL_SERVICE_AUTHORIZATION_HEADER] = (
            f"Bearer {inv_guardrails_apikey}"
        )

    class MockRequest:
        def __init__(self, headers):
            self.headers = headers

    dataset_name = "test-dataset"
    request = MockRequest(headers)

    try:
        invariant_authorization, llm_provider_api_key = (
            extract_authorization_from_headers(
                request,
                dataset_name=dataset_name,
                llm_provider_api_key_header="Authorization",
            )
        )

        context = RequestContext.create(
            request_json={"input": "test"},
            dataset_name=dataset_name,
            invariant_authorization=invariant_authorization,
            guardrails=None,
            config=GatewayConfig(),
            request=request,
        )
    except HTTPException as e:
        # If an exception is raised, check if it is the expected one
        if not invariant_authorization:
            assert e.status_code == 400
            assert e.detail == "Missing invariant api key"
            return
        else:
            raise e

    # Verify the results
    assert invariant_authorization == ("Bearer " + inv_apikey)
    assert llm_provider_api_key == llm_apikey
    assert context.get_guardrailing_authorization() == (
        "Bearer " + inv_guardrails_apikey
        if use_guardrailing_api_key
        else invariant_authorization
    )

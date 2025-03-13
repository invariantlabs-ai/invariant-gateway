"""Util functions for tests"""

import os

import pytest

@pytest.fixture
def gateway_url():
    """Get the gateway URL from the environment variable"""
    if "INVARIANT_GATEWAY_API_URL" in os.environ:
        return os.environ["INVARIANT_GATEWAY_API_URL"]
    raise ValueError("Please set the INVARIANT_GATEWAY_API_URL environment variable")


@pytest.fixture
def explorer_api_url():
    """Get the explorer API URL from the environment variable"""
    if "INVARIANT_API_URL" in os.environ:
        return os.environ["INVARIANT_API_URL"]
    raise ValueError("Please set the INVARIANT_API_URL environment variable")

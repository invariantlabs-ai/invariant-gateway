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


@pytest.fixture
def invariant_gateway_package_whl_file():
    """Get the Invariant Gateway package wheel file"""
    whl_file = None
    for filename in os.listdir("/package"):
        if filename.endswith(".whl") and "invariant_gateway" in filename:
            whl_file = filename
            break

    if whl_file:
        return f"/package/{whl_file}"
    raise ValueError("No Invariant Gateway wheel file found in /package")

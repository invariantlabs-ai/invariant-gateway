"""Util functions for tests"""

import os

import pytest
from playwright.async_api import async_playwright


@pytest.fixture
def proxy_url():
    if "INVARIANT_PROXY_API_URL" in os.environ:
        return os.environ["INVARIANT_PROXY_API_URL"]
    raise ValueError("Please set the INVARIANT_PROXY_API_URL environment variable")


@pytest.fixture
def explorer_api_url():
    if "INVARIANT_API_URL" in os.environ:
        return os.environ["INVARIANT_API_URL"]
    raise ValueError("Please set the INVARIANT_API_URL environment variable")


@pytest.fixture
async def playwright(scope="session"):
    async with async_playwright() as playwright_instance:
        yield playwright_instance


@pytest.fixture
async def browser(playwright, scope="session"):
    browser = await playwright.firefox.launch(headless=True)
    yield browser
    await browser.close()


@pytest.fixture
async def context(browser):
    context = await browser.new_context(ignore_https_errors=True)
    yield context
    await context.close()

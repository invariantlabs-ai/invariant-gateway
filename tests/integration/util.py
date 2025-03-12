"""Util functions for tests"""

import os

import pytest
from playwright.async_api import async_playwright


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
async def playwright(scope="session"):
    """Fixture to create a Playwright instance"""
    async with async_playwright() as playwright_instance:
        yield playwright_instance


@pytest.fixture
async def browser(playwright, scope="session"):
    """Fixture to create a browser instance"""
    firefox_browser = await playwright.firefox.launch(headless=True)
    yield firefox_browser
    await firefox_browser.close()


@pytest.fixture
async def context(browser):
    """Fixture to create a browser context"""
    browser_context = await browser.new_context(ignore_https_errors=True)
    yield browser_context
    await browser_context.close()

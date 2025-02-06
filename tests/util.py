"""Util functions for tests"""

import pytest
from playwright.async_api import async_playwright


@pytest.fixture
def url():
    return "http://127.0.0.1"


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

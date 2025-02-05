"""Test the chat completions proxy calls without tool calling."""

import os

# add tests folder (parent) to sys.path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import *  # needed for pytest fixtures

pytest_plugins = ("pytest_asyncio",)


async def test_hello_world(context, url):
    """Demo test"""
    response = await context.request.get(
        f"{url}/api/v1/dataset/byuser/developer/Welcome-to-Explorer"
    )
    dataset = await response.json()
    assert dataset["name"] == "Welcome-to-Explorer"
    assert dataset["user"]["username"] == "developer"

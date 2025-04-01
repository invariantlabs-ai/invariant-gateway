"""Common utilities for integration tests."""

import os
from httpx import Client
from openai import OpenAI
from google import genai
from anthropic import Anthropic


def get_open_ai_client(
    gateway_url: str, push_to_explorer: bool, dataset_name: str
) -> OpenAI:
    """Create an OpenAI client for integration tests."""
    return OpenAI(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/openai"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/openai",
    )


def get_anthropic_client(
    gateway_url: str, push_to_explorer: bool, dataset_name: str
) -> Anthropic:
    """Create an Anthropic client for integration tests."""
    return Anthropic(
        http_client=Client(
            headers={
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },
        ),
        base_url=f"{gateway_url}/api/v1/gateway/{dataset_name}/anthropic"
        if push_to_explorer
        else f"{gateway_url}/api/v1/gateway/anthropic",
    )


def get_gemini_client(
    gateway_url: str, push_to_explorer: bool, dataset_name: str
) -> genai.Client:
    """Create a Gemini client for integration tests."""
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={
            "base_url": f"{gateway_url}/api/v1/gateway/{dataset_name}/gemini"
            if push_to_explorer
            else f"{gateway_url}/api/v1/gateway/gemini",
            "headers": {
                "Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"
            },
        },
    )

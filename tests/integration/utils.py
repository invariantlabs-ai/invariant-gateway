"""Common utilities for integration tests."""

import os
import uuid
from typing import Any, Dict, Literal, Optional

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


async def create_dataset(
    explorer_api_url: str,
    invariant_authorization: str,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a dataset in the Explorer API."""
    client = Client(base_url=explorer_api_url)
    response = client.post(
        "/api/v1/dataset/create",
        json={"name": dataset_name if dataset_name else f"test-dataset-{uuid.uuid4()}"},
        headers={"Authorization": invariant_authorization},
        timeout=5,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to create dataset: {response.status_code}, {response.text}"
        )
    return response.json()


async def add_guardrail_to_dataset(
    explorer_api_url: str,
    dataset_id: str,
    policy: str,
    action: Literal["block", "log"],
    invariant_authorization: str,
) -> Dict[str, Any]:
    """Add a guardrail to a dataset."""
    client = Client(base_url=explorer_api_url)
    response = client.post(
        f"/api/v1/dataset/{dataset_id}/policy",
        json={
            "action": action,
            "policy": policy,
            "name": f"test-guardrail-{uuid.uuid4()}",
        },
        headers={"Authorization": invariant_authorization},
        timeout=5,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to add guardrail: {response.status_code}, {response.text}"
        )
    return response.json()

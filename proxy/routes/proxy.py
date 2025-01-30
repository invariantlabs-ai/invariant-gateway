"""Proxy service to forward requests to the appropriate language model provider"""

from enum import Enum

from fastapi import APIRouter, Depends, Header, HTTPException, Request

proxy = APIRouter()


class LLMProvider(str, Enum):
    """Supported language model providers"""

    OPEN_AI = "openai"
    ANTHROPIC = "anthropic"

    @classmethod
    def is_valid(cls, provider: str) -> bool:
        """Check if a provider is a valid LLM provider"""
        return provider in {provider.value for provider in cls}


def validate_headers(invariant_authorization: str = Header(None)):
    """Require the Invariant-Authorization header to be present"""
    if invariant_authorization is None:
        raise HTTPException(
            status_code=400, detail="Missing Invariant-Authorization header"
        )
    return invariant_authorization


@proxy.post(
    "/{username}/{dataset_name}/{llm_provider}",
    dependencies=[Depends(validate_headers)],
)
async def chat_completion(
    request: Request,
    username: str,
    dataset_name: str,
    llm_provider: str,
):
    """Proxy call to a language model provider"""

    if not LLMProvider.is_valid(llm_provider):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported LLM provider '{llm_provider}'.",
        )

    return {"message": f"Upload {dataset_name} for {username} to {llm_provider}"}

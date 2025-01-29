"""Proxy service to forward requests to the appropriate language model provider"""

from fastapi import FastAPI

proxy = FastAPI()


@proxy.post("/u/{username}/{dataset_name}/{llm_provider}")
async def chat_completion(username: str, dataset_name: str, llm_provider: str):
    """Proxy call to a language model provider"""
    return {"message": f"Upload {dataset_name} for {username} to {llm_provider}"}

"""Proxy service to forward requests to the appropriate language model provider"""

from fastapi import APIRouter, Depends, Header, HTTPException, Request

proxy = APIRouter()


def validate_headers(invariant_authorization: str = Header(None)):
    """Require the invariant-authorization header to be present"""
    if invariant_authorization is None:
        raise HTTPException(
            status_code=400, detail="Missing invariant-authorization header"
        )
    return invariant_authorization


allowed_openai_endpoints = {"chat/completions", "moderations"}


@proxy.post(
    "/{username}/{dataset_name}/openai/{endpoint:path}",
    dependencies=[Depends(validate_headers)],
)
async def openai_proxy(
    request: Request,
    username: str,
    dataset_name: str,
    endpoint: str,
):
    """Proxy call to a language model provider"""
    print("headers: ", dict(request.headers))
    # print("request: ", await request.json())
    print(f"Proxying to OpenAI endpoint: {endpoint}")
    if endpoint not in allowed_openai_endpoints:
        raise HTTPException(status_code=404, detail="Not supported OpenAI endpoint")
    return {"message": f"Upload {dataset_name} for {username} to openai"}

"""Proxy service to forward requests to the Anthropic APIs"""

from fastapi import APIRouter, Header, HTTPException, Depends, Request

proxy = APIRouter()

ALLOWED_ANTHROPIC_ENDPOINTS = {"v1/messages"}
MISSING_INVARIANT_AUTH_HEADER = "Missing invariant-authorization header"
MISSING_AUTH_HEADER = "Missing authorization header"
NOT_SUPPORTED_ENDPOINT = "Not supported OpenAI endpoint"
FAILED_TO_PUSH_TRACE = "Failed to push trace to the dataset: "


def validate_headers(
        invariant_authorization: str = Header(None), authorization: str = Header(None)
):
    """Require the invariant-authorization and authorization headers to be present"""
    if invariant_authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_INVARIANT_AUTH_HEADER)
    # if authorization is None:
    #     raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)
    
@proxy.post(
    "/{dataset_name}/anthropic/{endpoint:path}",
    dependencies=[Depends(validate_headers)],
) 
async def anthropic_proxy(
    dataset_name: str,
    endpoint: str,
    request: Request,
):
    """Proxy calls to the Anthropic APIs"""
    if endpoint not in ALLOWED_ANTHROPIC_ENDPOINTS:
        raise HTTPException(status_code=404, detail=NOT_SUPPORTED_ENDPOINT)

    headers = {
        k: v for k, v in request.headers.items()
    }
    headers["accept-encoding"] = "identity"

    request_body = await request.body()

    print("request_body", request_body)

"""Proxy service to forward requests to the appropriate language model provider"""

import gzip
import json
from io import BytesIO

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response

ALLOWED_OPEN_AI_ENDPOINTS = {"chat/completions", "moderations"}
IGNORED_HEADERS = [
    "accept-encoding",
    "host",
    "invariant-authorization",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
    "x-real-ip",
]
proxy = APIRouter()


def validate_headers(
    invariant_authorization: str = Header(None), authorization: str = Header(None)
):
    """Require the invariant-authorization and authorization headers to be present"""
    if invariant_authorization is None:
        raise HTTPException(
            status_code=400, detail="Missing invariant-authorization header"
        )
    if authorization is None:
        raise HTTPException(status_code=400, detail="Missing authorization header")


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
    if endpoint not in ALLOWED_OPEN_AI_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not supported OpenAI endpoint")

    headers = dict(request.headers)
    print("🔹 Original Headers:", json.dumps(headers, indent=2))

    # Remove extra headers
    for h in IGNORED_HEADERS:
        headers.pop(h, None)
    headers["accept-encoding"] = "identity"

    body_bytes = await request.body()

    async with httpx.AsyncClient() as client:
        open_ai_request = client.build_request(
            "POST",
            f"https://api.openai.com/v1/{endpoint}",
            content=body_bytes,
            headers=headers,
        )
        print("🔹 Forwarded Headers:", json.dumps(headers, indent=2))
        response = await client.send(open_ai_request)
        # Log response details
        print(f"⬅️ Response Status: {response.status_code}")
        print(f"⬅️ Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        raw_response = response.content

        # Detect if original request expects gzip encoding
        original_accept_encoding = request.headers.get("accept-encoding", "")
        should_gzip = "gzip" in original_accept_encoding.lower()

        if should_gzip:
            # Compress the response using gzip
            gzip_buffer = BytesIO()
            with gzip.GzipFile(mode="wb", fileobj=gzip_buffer) as gz:
                gz.write(raw_response)
            compressed_response = gzip_buffer.getvalue()

            response_headers = dict(response.headers)
            response_headers["Content-Encoding"] = "gzip"
            response_headers["Content-Length"] = str(len(compressed_response))

            return Response(
                content=compressed_response,
                status_code=response.status_code,
                headers=response_headers,
            )
        return Response(
            content=raw_response,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

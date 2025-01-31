"""Proxy service to forward requests to the OpenAI APIs"""

import gzip
import json
from io import BytesIO

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from utils.explorer import push_trace

ALLOWED_OPEN_AI_ENDPOINTS = {"chat/completions"}
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
    if authorization is None:
        raise HTTPException(status_code=400, detail=MISSING_AUTH_HEADER)


@proxy.post(
    "/{dataset_name}/openai/{endpoint:path}",
    dependencies=[Depends(validate_headers)],
)
async def openai_proxy(
    request: Request,
    dataset_name: str,
    endpoint: str,
):
    """Proxy calls to the OpenAI APIs"""
    if endpoint not in ALLOWED_OPEN_AI_ENDPOINTS:
        raise HTTPException(status_code=404, detail=NOT_SUPPORTED_ENDPOINT)

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    request_body = await request.body()

    async with httpx.AsyncClient() as client:
        open_ai_request = client.build_request(
            "POST",
            f"https://api.openai.com/v1/{endpoint}",
            content=request_body,
            headers=headers,
        )
        response = await client.send(open_ai_request)
        try:
            json_response = response.json()
            # push messages to the Invariant Explorer
            # use both the request and response messages
            messages = json.loads(request_body).get("messages", [])
            messages += [
                choice["message"] for choice in json_response.get("choices", [])
            ]
            _ = push_trace(
                dataset_name=dataset_name,
                messages=[messages],
                invariant_authorization=request.headers.get("invariant-authorization"),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=FAILED_TO_PUSH_TRACE + str(e)
            ) from e

        # Detect if original request expects gzip encoding
        if "gzip" in request.headers.get("accept-encoding", "").lower():
            # Compress the response using gzip
            gzip_buffer = BytesIO()
            with gzip.GzipFile(mode="wb", fileobj=gzip_buffer) as gz:
                gz.write(response.content)
            compressed_response = gzip_buffer.getvalue()

            response_headers = dict(response.headers)
            response_headers.pop("Content-Encoding", None)
            response_headers.pop("Content-Length", None)
            response_headers["Content-Encoding"] = "gzip"
            response_headers["Content-Length"] = str(len(compressed_response))

            return Response(
                content=compressed_response,
                status_code=response.status_code,
                headers=response_headers,
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

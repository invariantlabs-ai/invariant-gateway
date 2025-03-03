"""Proxy service to forward requests to the Gemini APIs"""

import json

from common.config_manager import ProxyConfig, ProxyConfigManager
from fastapi import APIRouter, Depends, Request, Response
from utils.constants import IGNORED_HEADERS

proxy = APIRouter()


def _extract_dataset_name_and_endpoint(endpoint: str):
    """Extracts the dataset name and endpoint from the given endpoint."""
    endpoint_parts = endpoint.split("/")
    dataset_name = None
    if endpoint_parts[1] == "models":
        # Case 1: Without dataset_name
        # `endpoint = <version>/models/<model-name>:generateContent`
        reconstructed_endpoint = "/".join(endpoint_parts)
    elif endpoint_parts[2] == "models":
        # Case 2: With dataset_name
        # `endpoint = <dataset-name>/<version>/models/<model-name>:generateContent`
        dataset_name = endpoint_parts[0]
        reconstructed_endpoint = "/".join(endpoint_parts[1:])
    else:
        # Case 3: Invalid endpoint
        return Response(
            content=f"Invalid endpoint: {endpoint} - the endpoint should be in the format: \
            /api/v1/proxy/gemini/<version>/models/<model-name>:generateContent or \
            /api/v1/proxy/gemini/<dataset-name>/<version>models/<model-name>:generateContent",
            status_code=400,
        )
    return dataset_name, reconstructed_endpoint


@proxy.post(
    "/gemini/{endpoint:path}",
)
async def gemini_generate_content_proxy(
    request: Request,
    endpoint: str,
    config: ProxyConfig = Depends(ProxyConfigManager.get_config),  # pylint: disable=unused-argument
) -> Response:
    """Proxy calls to the OpenAI APIs"""
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in IGNORED_HEADERS
    }
    headers["accept-encoding"] = "identity"

    request_body_bytes = await request.body()
    request_body_json = json.loads(request_body_bytes)
    api_key = headers.get("x-goog-api-key")
    print(f"API Key: {api_key}")
    print("request body json: ", request_body_json)
    dataset_name, reconstructed_endpoint = _extract_dataset_name_and_endpoint(endpoint)

    print(f"API Key: {api_key}")
    print("Processed Endpoint: ", reconstructed_endpoint)
    print("Dataset Name: ", dataset_name)

    return {}

"""Serve the API"""

import fastapi
import uvicorn
from starlette_compress import CompressMiddleware

from gateway.routes.anthropic import gateway as anthropic_gateway
from gateway.routes.gemini import gateway as gemini_gateway
from gateway.routes.open_ai import gateway as open_ai_gateway
from gateway.mcp.sse import gateway as mcp_sse_gateway
from gateway.mcp.streamable import gateway as mcp_streamable_gateway

app = fastapi.app = fastapi.FastAPI(
    docs_url="/api/v1/gateway/docs",
    redoc_url="/api/v1/gateway/redoc",
    openapi_url="/api/v1/gateway/openapi.json",
)
app.add_middleware(CompressMiddleware)

router = fastapi.APIRouter(prefix="/api/v1")


@router.get("/gateway/health", tags=["health_check"], include_in_schema=False)
async def check_health():
    """Health check"""
    return {"message": "Hello from Invariant v1/gateway"}


router.include_router(open_ai_gateway, prefix="/gateway", tags=["open_ai_gateway"])

router.include_router(anthropic_gateway, prefix="/gateway", tags=["anthropic_gateway"])

router.include_router(gemini_gateway, prefix="/gateway", tags=["gemini_gateway"])

router.include_router(mcp_sse_gateway, prefix="/gateway", tags=["mcp_sse_gateway"])

router.include_router(
    mcp_streamable_gateway, prefix="/gateway", tags=["mcp_streamable_gateway"]
)

app.include_router(router)


# on / redirect to https://explorer.invariantlabs.ai/docs/gateway
@app.get("/", include_in_schema=False)
async def redirect_to_explorer():
    """Redirect to the explorer"""
    return fastapi.responses.RedirectResponse(
        url="https://explorer.invariantlabs.ai/docs/gateway"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

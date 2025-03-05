"""Serve the API"""

import fastapi
import uvicorn
from routes.anthropic import gateway as anthropic_gateway
from routes.gemini import gateway as gemini_gateway
from routes.open_ai import gateway as open_ai_gateway
from starlette_compress import CompressMiddleware

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

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

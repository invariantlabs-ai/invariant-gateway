"""Serve the API"""

import fastapi
import uvicorn
from routes.anthropic import proxy as anthropic_proxy
from routes.open_ai import proxy as open_ai_proxy
from starlette_compress import CompressMiddleware

app = fastapi.app = fastapi.FastAPI(
    docs_url="/api/v1/proxy/docs",
    redoc_url="/api/v1/proxy/redoc",
    openapi_url="/api/v1/proxy/openapi.json",
)
app.add_middleware(CompressMiddleware)

router = fastapi.APIRouter(prefix="/api/v1")


@router.get("/proxy/health", tags=["health_check"], include_in_schema=False)
async def get_proxy_info():
    """Health check"""
    return {"message": "Hello from Invariant v1/proxy"}


router.include_router(open_ai_proxy, prefix="/proxy", tags=["open_ai_proxy"])

router.include_router(anthropic_proxy, prefix="/proxy", tags=["anthropic_proxy"])

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

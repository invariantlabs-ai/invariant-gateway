"""Serve the API"""

import fastapi
import uvicorn
from routes.anthropic import proxy as anthropic_proxy
from routes.open_ai import proxy as open_ai_proxy

app = fastapi.FastAPI()

router = fastapi.APIRouter(prefix="/api/v1")

router.include_router(open_ai_proxy, prefix="/proxy", tags=["open_ai_proxy"])

router.include_router(anthropic_proxy, prefix="/proxy", tags=["anthropic_proxy"])

app.include_router(router)

# Serve the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

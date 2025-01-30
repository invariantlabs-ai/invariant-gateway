"""Serve the API"""

import fastapi
import uvicorn
from routes.proxy import proxy

app = fastapi.FastAPI()

router = fastapi.APIRouter(prefix="/api/v1")

router.include_router(proxy, prefix="/proxy", tags=["proxy"])

app.include_router(router)

# Serve the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

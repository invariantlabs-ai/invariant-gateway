"""Serve the API"""

import fastapi
from proxy import proxy

v1 = fastapi.FastAPI()

# install the API routes
v1.mount("/proxy", proxy)


# for debugging, we can check if the API is up
@v1.get("/")
async def home():
    """Check if the API is up"""
    return {"message": "Hello v1 proxy"}


# mount the API under /api/v1
proxy_app = fastapi.FastAPI()
proxy_app.mount("/api/v1", v1)

# serve the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(proxy_app, host="0.0.0.0", port=8000)

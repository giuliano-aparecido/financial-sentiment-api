from fastapi import Header, HTTPException

from app.config import API_KEY


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

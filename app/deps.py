import secrets

from fastapi import Header, HTTPException, Query

from app.config import API_KEY, DEFAULT_MODEL
from app.services.inference import resolve_model


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not API_KEY or not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def selected_model(
    model: str = Query(DEFAULT_MODEL, description="A configured model name - GET /health lists them"),
) -> str:
    try:
        return resolve_model(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

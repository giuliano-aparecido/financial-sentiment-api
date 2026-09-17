from fastapi import APIRouter

from app.config import DEFAULT_MODEL
from app.services.inference import inference_urls

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "default_model": DEFAULT_MODEL,
        "models": sorted(inference_urls()),
    }

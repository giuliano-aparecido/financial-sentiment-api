from fastapi import APIRouter

from app.config import DEFAULT_MODEL, HF_MODEL_REPO
from app.services.inference import inference_urls

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "default_model": DEFAULT_MODEL,
        "models": sorted(inference_urls()),
        "target_model_repo": HF_MODEL_REPO,
    }

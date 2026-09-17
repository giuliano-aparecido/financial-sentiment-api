from fastapi import APIRouter

from app.config import DEFAULT_MODEL, HF_MODEL_REPO, SUPPORTED_MODELS
from app.services.inference import configured_inference_url

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "default_model": DEFAULT_MODEL,
        "models": {name: bool(configured_inference_url(name)) for name in SUPPORTED_MODELS},
        "target_model_repo": HF_MODEL_REPO,
    }

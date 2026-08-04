from fastapi import APIRouter

from app.config import HF_MODEL_REPO, MODEL_ARCHITECTURE

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "active_architecture": MODEL_ARCHITECTURE,
        "target_model_repo": HF_MODEL_REPO,
    }

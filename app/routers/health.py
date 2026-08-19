from fastapi import APIRouter

from app.config import HF_MODEL_REPO, MODEL_ARCHITECTURE

router = APIRouter()


# GET and HEAD both, not just GET: confirmed live - UptimeRobot's keep-
# warm ping (see the fleet-wide "keep Render warm" history - git log:
# ci/remove-keep-alive-workflow) sends HEAD requests here, and this
# FastAPI/Starlette version does NOT auto-add HEAD support to a route
# declared with only @router.get - a HEAD request was getting a genuine
# 405. Doesn't affect Render's own keep-warm behavior (any request,
# regardless of status code, resets its idle timer), but it does mean
# UptimeRobot's own monitor was checking a method this endpoint didn't
# actually support.
@router.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "online",
        "active_architecture": MODEL_ARCHITECTURE,
        "target_model_repo": HF_MODEL_REPO,
    }

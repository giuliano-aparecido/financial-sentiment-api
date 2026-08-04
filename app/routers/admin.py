import logging
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi.util import get_remote_address

from app.config import ALLOWED_INFERENCE_HOST_SUFFIXES
from app.deps import verify_api_key
from app.models import UpdateInferenceURLRequest
from app.services.inference import get_hf_inference_url, set_hf_inference_url

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/update-inference-url", dependencies=[Depends(verify_api_key)])
async def update_inference_url(req: UpdateInferenceURLRequest, request: Request):
    caller_ip = get_remote_address(request)
    new_url = req.url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(new_url)

    # Anyone holding the API key could otherwise repoint this at an internal
    # address or cloud metadata endpoint - analyze_stock then POSTs the HF
    # bearer token there on every request, so this isn't just a bad-config
    # risk, it's a token-exfiltration one.
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host.endswith(ALLOWED_INFERENCE_HOST_SUFFIXES):
        logger.warning("update-inference-url REJECTED from %s: %r (disallowed scheme/host)", caller_ip, new_url)
        raise HTTPException(
            status_code=400,
            detail="url must be https and match an allowed host (see ALLOWED_INFERENCE_HOST_SUFFIXES)",
        )

    old_url = get_hf_inference_url()
    set_hf_inference_url(new_url)
    logger.info("update-inference-url OK from %s: %r -> %r", caller_ip, old_url, new_url)
    return {"status": "ok", "hf_inference_url": get_hf_inference_url()}

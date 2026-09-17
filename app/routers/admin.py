import logging
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi.util import get_remote_address

from app.config import DEFAULT_MODEL, is_allowed_inference_host, normalize_model_name
from app.deps import verify_api_key
from app.models import UpdateInferenceURLRequest, UpdateYfCrumbRequest
from app.services.inference import inference_urls, set_inference_url
from app.services.swiss_universe import reseed_yf_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/update-inference-url", dependencies=[Depends(verify_api_key)])
async def update_inference_url(req: UpdateInferenceURLRequest, request: Request):
    caller_ip = get_remote_address(request)
    try:
        model = normalize_model_name(req.model or DEFAULT_MODEL)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    new_url = req.url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(new_url)

    # Anyone holding the API key could otherwise repoint this at an internal
    # address or cloud metadata endpoint - analyze_stock then POSTs the HF
    # bearer token there on every request, so this isn't just a bad-config
    # risk, it's a token-exfiltration one.
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not is_allowed_inference_host(host):
        logger.warning("update-inference-url REJECTED from %s for [%s]: %r (disallowed scheme/host)", caller_ip, model, new_url)
        raise HTTPException(
            status_code=400,
            detail="url must be https and match an allowed host (see ALLOWED_INFERENCE_HOST_SUFFIXES)",
        )

    old_url = inference_urls().get(model)
    set_inference_url(new_url, model)
    logger.info("update-inference-url OK from %s for [%s]: %r -> %r", caller_ip, model, old_url, new_url)
    return {"status": "ok", "model": model, "hf_inference_url": new_url}


@router.post("/api/update-yf-crumb", dependencies=[Depends(verify_api_key)])
async def update_yf_crumb(req: UpdateYfCrumbRequest, request: Request):
    """Hot-swaps the running process's yfinance crumb/cookie pair with a
    freshly captured one - see swiss_universe.reseed_yf_session's own
    docstring. Called by scripts/refresh_yf_crumb.py instead of that
    script's old Render-env-var-push-plus-redeploy path, added 2026-08-20
    at the user's explicit request to stop redeploying the whole app just
    to refresh a crumb. Gated behind the same API key as everything else
    here - same rationale as update-inference-url: this is a privileged
    write into process state, not a public read."""
    caller_ip = get_remote_address(request)
    reseed_yf_session(req.crumb, req.cookies)
    logger.info("update-yf-crumb OK from %s (crumb len=%d, %d cookies)", caller_ip, len(req.crumb), len(req.cookies))
    return {"status": "ok"}

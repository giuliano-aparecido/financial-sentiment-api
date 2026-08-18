from fastapi import APIRouter, Depends, HTTPException, Request

from app.deps import verify_api_key
from app.limiter import limiter
from app.services.research_job import get_indicator_status, get_status, start_indicator_scan, start_scan
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS

router = APIRouter()


@router.post("/api/research/volatility/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("1/5minutes")
async def start_volatility_scan(request: Request):
    # request is required (unused directly) - slowapi's @limiter.limit
    # inspects the endpoint signature for a Request param to extract the
    # rate-limit key from (see app/limiter.py's rate_limit_key).
    #
    # Deliberately far stricter than this app's normal 10/minute default
    # (see app/limiter.py): a single scan makes ~150+ live calls to
    # Yahoo's unofficial endpoints, so repeated triggering risks getting
    # this server's IP rate-limited/blocked by Yahoo, which would break
    # this AND the main /api/analyze flow for everyone. start_scan()
    # itself is also idempotent while a scan is in flight (see
    # research_job.py), so this limit is a backstop against genuinely
    # repeated NEW scans, not against polling for status.
    return start_scan()


@router.get("/api/research/volatility/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def volatility_scan_status(request: Request):
    # Cheap (reads an in-memory dict, no yfinance calls) - overridden to a
    # much higher limit than the 10/minute default specifically so the
    # frontend can poll this every few seconds while a scan is running
    # without tripping the same budget sized for expensive requests.
    return get_status()


@router.post("/api/research/volatility/indicator/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("1/5minutes")
async def start_volatility_indicator_scan(request: Request, threshold_pct: float):
    # Entirely separate job from /start above (see research_job.py's
    # module docstring) - this table has its own independent Refresh
    # button + threshold selector in the frontend, so it must not share a
    # rate-limit bucket, job slot, or cache with the main scan; sharing
    # any of those would mean clicking this table's Refresh could block
    # on (or get silently ignored because of) the OTHER table's in-flight
    # scan, which defeats the whole point of it being independent.
    #
    # threshold_pct is REQUIRED (no default) - unlike the old all_caps
    # toggle, there's no sane "just pick one" default for a value that
    # changes what's actually being measured; the frontend's selector
    # always sends one explicitly.
    #
    # Validated against ALLOWED_THRESHOLD_PCTS server-side, not just
    # trusted from the request - the frontend's <select> only ever offers
    # these three values, but this endpoint still triggers a real
    # ~150-call live scan for whatever value it's given, so there's no
    # reason to accept one the UI never actually offers.
    if threshold_pct not in ALLOWED_THRESHOLD_PCTS:
        raise HTTPException(
            status_code=422,
            detail=f"threshold_pct must be one of {ALLOWED_THRESHOLD_PCTS}, got {threshold_pct}",
        )
    return start_indicator_scan(threshold_pct)


@router.get("/api/research/volatility/indicator/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def volatility_indicator_scan_status(request: Request):
    return get_indicator_status()

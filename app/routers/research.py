from fastapi import APIRouter, Depends, HTTPException, Request

from app.deps import verify_api_key
from app.limiter import limiter
from app.services.research_job import (
    get_indicator_status,
    get_rebound_status,
    get_today_status,
    start_indicator_scan,
    start_rebound_scan,
    start_today_scan,
)
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS

router = APIRouter()

# No @limiter.limit(...) override on any /start route below as of
# 2026-08-19 (previously "1/5minutes" on each) - removed at the user's
# explicit request: "user can start a new scan as long there is no scan
# of it running or the threshold was changed" - a flat time-window
# cooldown was blocking exactly that (a genuinely new, nothing-currently-
# running request), which defeats the point of splitting these into
# independent per-table buttons in the first place. The real protection
# against wasted duplicate yfinance calls is research_job.py's own
# single-flight-per-scan-type behavior (calling .../start while that
# SAME scan type is already running just returns its in-flight status,
# no new scan actually starts) - a time-based cooldown was ADDITIONAL
# friction on top of that, not the only thing preventing duplicate work.
# Each route still falls back to app.limiter's app-wide 10/minute default
# (see that module's own comment) as a loose backstop against a
# scripted/accidental burst, just no longer a business-logic-motivated
# stricter one.


@router.post("/api/research/volatility/rebound/start", dependencies=[Depends(verify_api_key)])
async def start_rebound_volatility_scan(request: Request):
    # request is required (unused directly) even with no @limiter.limit
    # override here - kept for signature consistency with the other
    # routes below and in case a future change reintroduces one.
    return start_rebound_scan()


@router.get("/api/research/volatility/rebound/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def rebound_volatility_scan_status(request: Request):
    # Cheap (reads an in-memory dict, no yfinance calls) - overridden to a
    # much higher limit than the 10/minute default specifically so the
    # frontend can poll this every few seconds while a scan is running
    # without tripping the same budget sized for expensive requests.
    return get_rebound_status()


@router.post("/api/research/volatility/today/start", dependencies=[Depends(verify_api_key)])
async def start_today_volatility_scan(request: Request):
    return start_today_scan()


@router.get("/api/research/volatility/today/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def today_volatility_scan_status(request: Request):
    return get_today_status()


@router.post("/api/research/volatility/indicator/start", dependencies=[Depends(verify_api_key)])
async def start_volatility_indicator_scan(request: Request, threshold_pct: float):
    # threshold_pct is REQUIRED (no default) - there's no sane "just pick
    # one" default for a value that changes what's actually being
    # measured; the frontend's selector always sends one explicitly.
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

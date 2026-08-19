import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from app.deps import verify_api_key
from app.limiter import limiter
from app.services import scan_persistence, scheduler
from app.services.research_job import get_today_status, start_today_scan
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS

router = APIRouter()

# --- Scheduled-scan reads + manual trigger (rebound, volatility-indicator) ---
#
# These two tables are no longer scanned live on every request - see
# app/services/scheduler.py's own docstring for why (user's explicit
# request: cache daily/monthly instead of hammering Yahoo on every button
# click). The GET routes below just read whatever the scheduler most
# recently persisted (app/services/scan_persistence.py) - cheap (one
# indexed DB query) - plus whether a scan is CURRENTLY running, since the
# user explicitly wants the frontend to show that state (button disabled,
# no table) regardless of whether that in-progress scan was started by
# the schedule or by a manual Refresh click. The POST .../start routes
# trigger a manual run through the EXACT SAME guarded pipeline the cron
# uses (scheduler.trigger_rebound_scan/trigger_indicator_scan) - single-
# flight per table, a click while one's already running is a no-op, not
# an error. "Today" (big-loss) below is untouched - stays fully live/on-
# demand via research_job.py's own job-slot machinery, since intraday
# data has no meaningful cache window.


@router.get("/api/research/volatility/rebound", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_rebound_scan_result(request: Request):
    rows, scan_run_at, failed_tickers = await asyncio.to_thread(scan_persistence.get_latest_rebound_scan)
    return {
        "rows": rows,
        "scan_run_at": scan_run_at.isoformat() if scan_run_at else None,
        "is_running": scheduler.is_rebound_scan_running(),
        "failed_ticker_count": len(failed_tickers),
    }


@router.post("/api/research/volatility/rebound/start", dependencies=[Depends(verify_api_key)])
async def start_rebound_volatility_scan(request: Request):
    started = await asyncio.to_thread(scheduler.trigger_rebound_scan)
    return {"started": started, "is_running": True}


@router.get("/api/research/volatility/indicator", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_indicator_scan_result(request: Request, threshold_pct: float):
    if threshold_pct not in ALLOWED_THRESHOLD_PCTS:
        raise HTTPException(
            status_code=422,
            detail=f"threshold_pct must be one of {ALLOWED_THRESHOLD_PCTS}, got {threshold_pct}",
        )
    rows, scan_run_at, failed_tickers = await asyncio.to_thread(scan_persistence.get_latest_indicator_scan, threshold_pct)
    return {
        "rows": rows,
        "scan_run_at": scan_run_at.isoformat() if scan_run_at else None,
        "is_running": scheduler.is_indicator_scan_running(),
        "failed_ticker_count": len(failed_tickers),
    }


@router.post("/api/research/volatility/indicator/start", dependencies=[Depends(verify_api_key)])
async def start_volatility_indicator_scan(request: Request, threshold_pct: float):
    # threshold_pct is still required/validated here even though the
    # triggered scan always covers every threshold (see scheduler.
    # trigger_indicator_scan) - keeps this endpoint's contract consistent
    # with the GET route above and rejects a malformed/tampered value
    # up front rather than silently ignoring it.
    if threshold_pct not in ALLOWED_THRESHOLD_PCTS:
        raise HTTPException(
            status_code=422,
            detail=f"threshold_pct must be one of {ALLOWED_THRESHOLD_PCTS}, got {threshold_pct}",
        )
    started = await asyncio.to_thread(scheduler.trigger_indicator_scan)
    return {"started": started, "is_running": True}


# No @limiter.limit(...) override on the /start routes above (rebound/
# indicator) or below (today) - previously "1/5minutes" on each, removed
# at the user's explicit request: "user can start a new scan as long
# there is no scan of it running or the threshold was changed" - a flat
# time-window cooldown was blocking exactly that (a genuinely new,
# nothing-currently-running request). The real protection against wasted
# duplicate yfinance calls is single-flight-per-scan-type behavior
# (scheduler.py's lock-guarded functions for rebound/indicator,
# research_job.py's _JobSlot for today) - calling .../start while that
# SAME scan type is already running just no-ops, no new scan actually
# starts. Each route still falls back to app.limiter's app-wide 10/minute
# default (see that module's own comment) as a loose backstop against a
# scripted/accidental burst, just no longer a business-logic-motivated
# stricter one.


@router.post("/api/research/volatility/today/start", dependencies=[Depends(verify_api_key)])
async def start_today_volatility_scan(request: Request):
    return start_today_scan()


@router.get("/api/research/volatility/today/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def today_volatility_scan_status(request: Request):
    return get_today_status()

from fastapi import APIRouter, Depends, Request

from app.deps import verify_api_key
from app.limiter import limiter
from app.services.research_job import get_status, start_scan

router = APIRouter()


@router.post("/api/research/small-caps/start", dependencies=[Depends(verify_api_key)])
@limiter.limit("1/5minutes")
async def start_small_caps_scan(request: Request):
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


@router.get("/api/research/small-caps/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def small_caps_scan_status(request: Request):
    # Cheap (reads an in-memory dict, no yfinance calls) - overridden to a
    # much higher limit than the 10/minute default specifically so the
    # frontend can poll this every few seconds while a scan is running
    # without tripping the same budget sized for expensive requests.
    return get_status()

"""
Background-job runner for the Swiss small-cap research scan
(swiss_small_cap_crash_rebound.py + swiss_small_cap_today_screener.py),
backing app/routers/research.py's start/status endpoints.

A full scan takes 1-3 minutes (universe discovery + ~100+ per-ticker
domicile checks + a batch price-history download), which is far past what
a single HTTP request should hold open - especially through
financial-sentiment-web's Vercel-hosted proxy, which has its own execution
time ceiling. So this runs the scan in a plain background thread, detached
from the request that started it: POST /start returns immediately with
"running", and the caller polls GET /status until it sees "done" or
"error". This is NOT the asyncio.to_thread pattern used elsewhere in this
app (see fundamentals.py/earnings.py) - to_thread AWAITS completion within
the same request/response cycle, which is exactly what would recreate the
timeout problem this is meant to avoid.

Single global job slot, not a job-id-keyed store: this app is a single
Render worker, and running two scans concurrently would just have them
contend for the same yfinance call budget for no benefit - there's nothing
to gain from tracking multiple simultaneous jobs for what is, in practice,
one person's research tool. If a scan is already running, starting a new
one just hands back the in-flight job's current status instead of queuing
or rejecting.

Module-level dict reassignment (`_job = {...}`) is used instead of mutating
a shared dict in place - each state transition is a single atomic pointer
swap, so a status poll reading `_job` while the background thread
"finishes" and reassigns it can't observe a half-written state, without
needing a lock around every read (CPython's GIL makes the rebind itself
atomic). The lock below only guards the "is one already running, if not
start one" check in start_scan(), which is the one place two threads could
otherwise race to both start a scan.

Crash-rebound caching: its 3-month lookback barely changes day to day -
once a trading day closes, that day's OHLCV doesn't change - so
_crash_rebound_result below only recomputes it (paying the yf.download
historical batch call) once per UTC calendar day, reusing the cached
result for same-day re-scans. today_screener is NEVER cached - it reports
live intraday quotes and always re-runs against the freshly-discovered
`domestic` dict. Since the only way to trigger a scan at all is the
Refresh button (no auto-poll timer), this already means today_screener's
result is "refreshed on demand via the button, or naturally on the next
day" - the exact behavior wanted, with no extra caching logic needed for
it specifically. What IS cached (the job's _job dict) just holds the most
recently computed result until the next scan overwrites it - not a TTL,
just normal state.

No lock needed around the crash-rebound cache: _run only ever executes
one at a time (see start_scan's single-job-slot reasoning above), so
there's no concurrent writer to race against.
"""

import datetime
import logging
import threading

import pandas as pd

from app.services import swiss_small_cap_crash_rebound, swiss_small_cap_today_screener, swiss_universe

logger = logging.getLogger(__name__)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list-of-dicts with NaN replaced by None. Confirmed
    live: df.to_dict(orient="records") alone leaves NaN (a real value in
    both scan modules - e.g. loss_pe_approx for a loss-making company, or
    volume_vs_3mo_avg when a ticker has no 3-month average yet) as the
    Python float nan, which Python's json.dumps happily renders as the
    bare token NaN - not valid JSON, and something JavaScript's
    JSON.parse (what financial-sentiment-web's proxy route eventually
    calls) throws a SyntaxError on. None -> JSON null is the only safe
    representation for "no value" crossing this boundary.
    """
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


_job: dict = {"status": "idle"}
_job_lock = threading.Lock()

_crash_rebound_cache: dict = {"date": None, "result": None}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _today() -> datetime.date:
    # Factored out (instead of inlining datetime.datetime.now(...).date()
    # in _crash_rebound_result below) so tests can monkeypatch "today"
    # directly rather than faking the datetime module. UTC, not CET/Swiss-
    # market-day - a simple daily boundary, not an exact trading-session
    # cutoff; a cache miss right at the UTC/CET offset just costs one
    # extra recompute, not a correctness problem.
    return datetime.datetime.now(datetime.timezone.utc).date()


def _crash_rebound_result(domestic: dict) -> pd.DataFrame:
    """Cached per calendar day (see module docstring) - returns today's
    already-computed result if this is a same-day re-scan, otherwise
    recomputes and caches it."""
    global _crash_rebound_cache
    today = _today()
    if _crash_rebound_cache["date"] == today:
        logger.info("Reusing cached crash-rebound result from %s", today)
        return _crash_rebound_cache["result"]
    result = swiss_small_cap_crash_rebound.run_scan(domestic)
    _crash_rebound_cache = {"date": today, "result": result}
    return result


def _run(started_at: str) -> None:
    global _job
    try:
        candidates = swiss_universe.discover_candidates()
        domestic = swiss_universe.filter_domestic(candidates)
        crash_rebound_df = _crash_rebound_result(domestic)
        today_df = swiss_small_cap_today_screener.run_scan(domestic)
        _job = {
            "status": "done",
            "started_at": started_at,
            "finished_at": _now(),
            "universe_size": len(domestic),
            "crash_rebound": _json_safe_records(crash_rebound_df),
            "today_screener": _json_safe_records(today_df),
        }
        logger.info(
            "Research scan done: universe=%d crash_rebound_matches=%d today_matches=%d",
            len(domestic), len(crash_rebound_df), len(today_df),
        )
    except Exception as e:
        logger.exception("Research scan failed")
        _job = {"status": "error", "started_at": started_at, "finished_at": _now(), "error": str(e)}


def start_scan() -> dict:
    """Starts a new scan if none is currently running; otherwise returns
    the already-in-flight job's current status unchanged. Either way,
    returns the same shape get_status() does, so callers (see
    app/routers/research.py) don't need two different response handlers
    for "just started" vs "already running"."""
    global _job
    with _job_lock:
        if _job.get("status") == "running":
            return dict(_job)
        started_at = _now()
        _job = {"status": "running", "started_at": started_at}
        threading.Thread(target=_run, args=(started_at,), daemon=True).start()
        return dict(_job)


def get_status() -> dict:
    return dict(_job)

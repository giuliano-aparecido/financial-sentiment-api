"""
Background-job runner for the Swiss volatility research scan
(swiss_crash_rebound.py + swiss_today_screener.py + swiss_volatility_
indicator.py), backing app/routers/research.py's start/status endpoints.

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

Universe: a single market-cap band (swiss_universe.MIN/MAX_MARKET_CAP_CHF,
CHF 500M+, no upper bound - SMI's 20 largest/most-liquid names still
EXCLUDED - see swiss_universe.SMI_TICKERS) as of 2026-08-18 - previously
two separate bands (a small-cap default plus an opt-in wider "all caps"
mode, threaded through this module via an `all_caps` parameter) collapsed
into one at the user's explicit request to always include mid/large caps.

Crash-rebound caching: its 12-month lookback barely changes day to day -
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

Volatility-indicator scan (swiss_volatility_indicator.py): a genuinely
separate job/cache/lock triple (_indicator_job/_indicator_cache/
_indicator_job_lock), not folded into _job above, because it has its own
independent trigger (a separate Refresh button + threshold selector in
the frontend - see start_indicator_scan) and must NOT block on or get
blocked by the main crash-rebound/today-screener scan. It also does its
OWN independent universe discovery rather than reusing _job's `domestic`
- see swiss_volatility_indicator.py's own module docstring for why.
Cached per (calendar day, threshold_pct) - same "a closed trading day's
OHLCV doesn't change" reasoning as _crash_rebound_cache, plus the
threshold in the key since 2%/3%/5% are genuinely different scans over
the same universe, not one scan with a display-only filter.
"""

import datetime
import logging
import threading

import pandas as pd

from app.services import swiss_crash_rebound, swiss_today_screener, swiss_universe, swiss_volatility_indicator

logger = logging.getLogger(__name__)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list-of-dicts with NaN replaced by None. Confirmed
    live: df.to_dict(orient="records") alone leaves NaN (a real value in
    both scan modules - e.g. loss_pe_approx for a loss-making company, or
    volume_vs_10d_avg when a ticker has no 10-day average yet) as the
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

_indicator_job: dict = {"status": "idle"}
_indicator_job_lock = threading.Lock()
_indicator_cache: dict = {"date": None, "threshold_pct": None, "result": None}


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
    """Cached per calendar day - returns today's already-computed result
    if this is a same-day re-scan, otherwise recomputes and caches it."""
    global _crash_rebound_cache
    today = _today()
    if _crash_rebound_cache["date"] == today:
        logger.info("Reusing cached crash-rebound result from %s", today)
        return _crash_rebound_cache["result"]
    result = swiss_crash_rebound.run_scan(domestic)
    _crash_rebound_cache = {"date": today, "result": result}
    return result


def _run(started_at: str) -> None:
    global _job
    try:
        candidates = swiss_universe.discover_candidates()
        domestic = swiss_universe.filter_domestic(candidates)
        crash_rebound_df = _crash_rebound_result(domestic)
        today_df = swiss_today_screener.run_scan(domestic)
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


def _indicator_result(domestic: dict, threshold_pct: float) -> pd.DataFrame:
    """Cached per (calendar day, threshold_pct) - see module docstring
    for why the threshold is part of the key."""
    global _indicator_cache
    today = _today()
    if _indicator_cache["date"] == today and _indicator_cache["threshold_pct"] == threshold_pct:
        logger.info("Reusing cached volatility-indicator result from %s (threshold=%s)", today, threshold_pct)
        return _indicator_cache["result"]
    result = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
    _indicator_cache = {"date": today, "threshold_pct": threshold_pct, "result": result}
    return result


def _run_indicator(started_at: str, threshold_pct: float) -> None:
    global _indicator_job
    try:
        # Independent discovery, not the main scan's `domestic` - see
        # swiss_volatility_indicator.py's own module docstring for why.
        candidates = swiss_universe.discover_candidates()
        domestic = swiss_universe.filter_domestic(candidates)
        indicator_df = _indicator_result(domestic, threshold_pct)
        _indicator_job = {
            "status": "done",
            "started_at": started_at,
            "finished_at": _now(),
            "threshold_pct": threshold_pct,
            "universe_size": len(domestic),
            "volatility_indicator": _json_safe_records(indicator_df),
        }
        logger.info(
            "Volatility-indicator scan done: threshold=%s universe=%d matches=%d",
            threshold_pct, len(domestic), len(indicator_df),
        )
    except Exception as e:
        logger.exception("Volatility-indicator scan failed")
        _indicator_job = {
            "status": "error", "started_at": started_at, "finished_at": _now(),
            "threshold_pct": threshold_pct, "error": str(e),
        }


def start_indicator_scan(threshold_pct: float) -> dict:
    """Starts a new volatility-indicator scan if none is currently
    running; otherwise returns the already-in-flight job's current status
    unchanged (including whatever threshold_pct that in-flight scan was
    started with - a second call with a different threshold while one is
    already running does NOT change or restart it, same idempotent-while-
    running behavior as start_scan). Entirely independent of start_scan/
    _job above - see module docstring."""
    global _indicator_job
    with _indicator_job_lock:
        if _indicator_job.get("status") == "running":
            return dict(_indicator_job)
        started_at = _now()
        _indicator_job = {"status": "running", "started_at": started_at, "threshold_pct": threshold_pct}
        threading.Thread(target=_run_indicator, args=(started_at, threshold_pct), daemon=True).start()
        return dict(_indicator_job)


def get_indicator_status() -> dict:
    return dict(_indicator_job)

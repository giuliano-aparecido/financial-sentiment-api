"""
Background-job runner for the Swiss volatility research scans
(swiss_crash_rebound.py + swiss_today_screener.py + swiss_volatility_
indicator.py), backing app/routers/research.py's start/status endpoints.

A full scan takes 1-3 minutes (universe discovery + ~100+ per-ticker
domicile checks + a batch price-history download), which is far past what
a single HTTP request should hold open - especially through
financial-sentiment-web's Vercel-hosted proxy, which has its own execution
time ceiling. So each scan runs in a plain background thread, detached
from the request that started it: POST .../start returns immediately with
"running", and the caller polls GET .../status until it sees "done" or
"error". This is NOT the asyncio.to_thread pattern used elsewhere in this
app (see fundamentals.py/earnings.py) - to_thread AWAITS completion within
the same request/response cycle, which is exactly what would recreate the
timeout problem this is meant to avoid.

THREE fully independent job slots as of 2026-08-19 (rebound/today/
indicator), not one combined slot - previously rebound and today shared
a single job/button (the OLD start_scan()/_job); split apart at the
user's explicit request so each table has its own Refresh button that
doesn't block on or get blocked by the others. Each does its OWN
independent universe discovery (discover_candidates()/filter_domestic())
rather than sharing one `domestic` dict across scan types - simpler and
safer than trying to invalidate a shared cache correctly across three
independently-triggered scans, at the cost of each button's own click
paying its own ~30-60s/~150-live-call discovery cost (same bounded,
user-initiated cost every one of these buttons already had before this
split - nothing NEW here, just no longer shared).

Single global job slot PER SCAN TYPE, not a job-id-keyed store: this app
is a single Render worker, and running two scans of the SAME type
concurrently would just have them contend for the same yfinance call
budget for no benefit - there's nothing to gain from tracking multiple
simultaneous same-type jobs for what is, in practice, one person's
research tool. If a scan of a given type is already running, starting a
new one of that SAME type just hands back the in-flight job's current
status instead of queuing, rejecting, or erroring - see _JobSlot.start.

No per-request/business-logic rate limiting at this layer (see
app/routers/research.py for why that was removed from the /start routes
2026-08-19) - the single-flight behavior above IS the protection against
wasted duplicate yfinance calls; a flat time-window cooldown was ALSO
blocking a genuinely new, cheap "nothing is running, start a fresh one"
request, which the user explicitly wants allowed (e.g. the volatility-
indicator table: switching threshold and immediately starting a new scan
for it, as long as nothing of that type is already in flight).

Module-level dict reassignment (`self.job = {...}`) inside _JobSlot is
used instead of mutating a shared dict in place - each state transition
is a single atomic pointer swap, so a status poll reading `self.job`
while the background thread "finishes" and reassigns it can't observe a
half-written state, without needing a lock around every read (CPython's
GIL makes the rebind itself atomic). The lock only guards the "is one
already running, if not start one" check in _JobSlot.start, which is the
one place two threads could otherwise race to both start a scan of the
same type.

Universe: a single market-cap band (swiss_universe.MIN/MAX_MARKET_CAP_CHF,
CHF 500M+, no upper bound - SMI's 20 largest/most-liquid names still
EXCLUDED - see swiss_universe.SMI_TICKERS) as of 2026-08-18.

Rebound caching: its 12-month lookback barely changes day to day - once a
trading day closes, that day's OHLCV doesn't change - so _rebound_result
below only recomputes it (paying the yf.download historical batch call)
once per UTC calendar day, reusing the cached result for same-day
re-scans (including across DIFFERENT users/browser tabs clicking
Refresh, or a fresh discovery pass with a different `domestic` object
but the same underlying trading data). today_screener's own SCAN RESULT
is never cached - it always recomputes fresh against whatever `domestic`
it's handed, so live intraday quotes stay live; volatility-indicator's
result is cached per (calendar day, threshold_pct), same "a closed
trading day's OHLCV doesn't change" reasoning, plus the threshold in the
key since 2%/3%/5% are genuinely different scans over the same universe,
not one scan with a display-only filter.

Discovery caching + partial-failure retry (added 2026-08-19): each scan
type ALSO caches its own `domestic` dict (and the `candidates` dict
discovery produced it from) per calendar day, via _discover_and_filter_
with_retry below - independent per scan type still (see the split
above), not shared across rebound/today/indicator. Confirmed live: some
tickers' .info fetch can fail transiently (Yahoo rate-limiting mid-scan)
while most succeed - swiss_universe.filter_domestic already failed soft
per-ticker (drops just that one, doesn't abort the scan), but the
dropped ticker was gone for good even though the SAME scan type's next
same-day click re-discovered the whole universe from scratch anyway,
paying the full ~150-call cost AGAIN just to end up with the same
partial result if Yahoo was still degraded, or a fresh full result that
silently discarded whatever succeeded the first time if only some
tickers were still failing. Now: a same-day re-scan reuses the cached
`domestic` dict outright and retries ONLY the specific tickers that
failed last time (their original discovery-time `quote` dict is kept in
`candidates` specifically so this retry doesn't need to re-run
discovery itself) - newly-succeeding tickers get merged permanently into
the day's cached `domestic`; still-failing ones stay in `failed_symbols`
for the NEXT same-day retry. `failed_ticker_count` is surfaced in each
scan's own "done"/"error" status so the frontend can show a warning when
a result may be incomplete rather than looking like a clean, complete
scan.

No lock needed around any of these caches: a given scan type's _run_*
only ever executes one at a time (see _JobSlot's single-flight
reasoning above), so there's no concurrent writer to race against.
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
    these scan modules - e.g. loss_pe_approx for a loss-making company)
    as the Python float nan, which Python's json.dumps happily renders as
    the bare token NaN - not valid JSON, and something JavaScript's
    JSON.parse (what financial-sentiment-web's proxy route eventually
    calls) throws a SyntaxError on. None -> JSON null is the only safe
    representation for "no value" crossing this boundary.
    """
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _today() -> datetime.date:
    # Factored out (instead of inlining datetime.datetime.now(...).date()
    # in the result-cache helpers below) so tests can monkeypatch "today"
    # directly rather than faking the datetime module. UTC, not CET/Swiss-
    # market-day - a simple daily boundary, not an exact trading-session
    # cutoff; a cache miss right at the UTC/CET offset just costs one
    # extra recompute, not a correctness problem.
    return datetime.datetime.now(datetime.timezone.utc).date()


class _JobSlot:
    """One independent background-job slot - single-flight (see module
    docstring), status readable at any time via .status(). run_fn is
    called as run_fn(slot, started_at, *run_args) in a daemon thread;
    it's responsible for calling slot.set_done(...)/slot.set_error(...)
    itself (rather than returning a value) so each scan type can attach
    whatever extra fields it needs (crash_rebound=..., threshold_pct=...,
    etc.) without this shared class needing to know their shape."""

    def __init__(self):
        self.job: dict = {"status": "idle"}
        self.lock = threading.Lock()

    def start(self, run_fn, *run_args, **running_fields) -> dict:
        with self.lock:
            if self.job.get("status") == "running":
                return dict(self.job)
            started_at = _now()
            self.job = {"status": "running", "started_at": started_at, **running_fields}
            threading.Thread(target=run_fn, args=(self, started_at, *run_args), daemon=True).start()
            return dict(self.job)

    def status(self) -> dict:
        return dict(self.job)

    def set_done(self, started_at: str, **fields) -> None:
        self.job = {"status": "done", "started_at": started_at, "finished_at": _now(), **fields}

    def set_error(self, started_at: str, error: str, **fields) -> None:
        self.job = {"status": "error", "started_at": started_at, "finished_at": _now(), "error": error, **fields}


def _new_discovery_cache() -> dict:
    return {"date": None, "candidates": {}, "domestic": {}, "failed_symbols": []}


def _discover_and_filter_with_retry(cache: dict) -> tuple[dict, bool]:
    """Per-scan-type discovery+domicile-filter, cached per calendar day
    with same-day partial-failure retry (see module docstring for the
    full "why"). `cache` is one of _rebound_discovery_cache/
    _today_discovery_cache/_indicator_discovery_cache - mutated in
    place. Returns (domestic, changed).

    `changed` is True whenever `domestic` is DIFFERENT from what it was
    on this cache's previous call (a fresh day's full discovery, or a
    same-day retry that actually recovered at least one previously-
    failed ticker) - callers MUST check this before trusting their OWN
    result-level cache (_rebound_result/_indicator_result): a scan
    result computed from a SMALLER `domestic` (because some tickers
    were still failing at the time) would otherwise keep being served
    as this scan type's "cached" result even after a later retry
    recovers those tickers, silently hiding the recovery. `changed` is
    False only for a genuine same-day no-op re-check (no failed_symbols
    left, or a retry that didn't recover anything new) - the one case
    where reusing an existing result-level cache is actually correct.

    Fresh day (or never run): full discover_candidates() + filter_
    domestic() pass, caches candidates/domestic/failed_symbols.

    Same day, with leftover failed_symbols from an earlier call today:
    retries ONLY those specific symbols (their original `quote` dict is
    still in `candidates`, so this doesn't need to re-run discovery) -
    newly-successful ones are merged into the cached `domestic`
    permanently for the rest of the day; still-failing ones remain in
    `failed_symbols` for the next call to retry again.

    Same day, no failed_symbols: pure cache hit, no yfinance calls at
    all - identical to the pre-2026-08-19 "cached once per day" result-
    level caching, just applied one layer earlier (to the universe
    itself, not just the final scan computation).
    """
    today = _today()
    if cache["date"] != today:
        candidates = swiss_universe.discover_candidates()
        domestic, failed_symbols = swiss_universe.filter_domestic(candidates)
        cache["date"] = today
        cache["candidates"] = candidates
        cache["domestic"] = domestic
        cache["failed_symbols"] = failed_symbols
        logger.info(
            "Discovery for %s: universe=%d failed=%d", today, len(domestic), len(failed_symbols),
        )
        return domestic, True

    if cache["failed_symbols"]:
        retry_candidates = {s: cache["candidates"][s] for s in cache["failed_symbols"] if s in cache["candidates"]}
        newly_domestic, still_failed = swiss_universe.filter_domestic(retry_candidates)
        cache["failed_symbols"] = still_failed
        logger.info(
            "Retried %d previously-failed tickers: %d now succeeded, %d still failing",
            len(retry_candidates), len(newly_domestic), len(still_failed),
        )
        if newly_domestic:
            cache["domestic"].update(newly_domestic)
            return cache["domestic"], True
    return cache["domestic"], False


# --- Rebound scan ---

_rebound_slot = _JobSlot()
_rebound_cache: dict = {"date": None, "result": None}
_rebound_discovery_cache: dict = _new_discovery_cache()


def _rebound_result(domestic: dict, force: bool) -> pd.DataFrame:
    """Cached per calendar day - returns today's already-computed result
    if this is a same-day re-scan AND `domestic` hasn't changed since
    (force=False - see _discover_and_filter_with_retry's own docstring
    for why `force` matters), otherwise recomputes and caches it."""
    global _rebound_cache
    today = _today()
    if not force and _rebound_cache["date"] == today:
        logger.info("Reusing cached rebound result from %s", today)
        return _rebound_cache["result"]
    result = swiss_crash_rebound.run_scan(domestic)
    _rebound_cache = {"date": today, "result": result}
    return result


def _run_rebound(slot: _JobSlot, started_at: str) -> None:
    try:
        domestic, changed = _discover_and_filter_with_retry(_rebound_discovery_cache)
        failed_count = len(_rebound_discovery_cache["failed_symbols"])
        df = _rebound_result(domestic, force=changed)
        slot.set_done(
            started_at, universe_size=len(domestic), failed_ticker_count=failed_count,
            crash_rebound=_json_safe_records(df),
        )
        logger.info(
            "Rebound scan done: universe=%d matches=%d failed=%d", len(domestic), len(df), failed_count,
        )
    except Exception as e:
        logger.exception("Rebound scan failed")
        slot.set_error(started_at, str(e))


def start_rebound_scan() -> dict:
    """Starts a new rebound scan if none is currently running; otherwise
    returns the already-in-flight job's current status unchanged."""
    return _rebound_slot.start(_run_rebound)


def get_rebound_status() -> dict:
    return _rebound_slot.status()


# --- Today (big-loss) scan ---

_today_slot = _JobSlot()
_today_discovery_cache: dict = _new_discovery_cache()


def _run_today(slot: _JobSlot, started_at: str) -> None:
    try:
        # `changed` unused here - today_screener's own scan RESULT is
        # never cached regardless (always live intraday quotes, see
        # module docstring), only the discovery/domestic step benefits
        # from the same-day cache-with-retry.
        domestic, _changed = _discover_and_filter_with_retry(_today_discovery_cache)
        failed_count = len(_today_discovery_cache["failed_symbols"])
        df = swiss_today_screener.run_scan(domestic)
        slot.set_done(
            started_at, universe_size=len(domestic), failed_ticker_count=failed_count,
            today_screener=_json_safe_records(df),
        )
        logger.info(
            "Today (big-loss) scan done: universe=%d matches=%d failed=%d", len(domestic), len(df), failed_count,
        )
    except Exception as e:
        logger.exception("Today (big-loss) scan failed")
        slot.set_error(started_at, str(e))


def start_today_scan() -> dict:
    """Starts a new today/big-loss scan if none is currently running;
    otherwise returns the already-in-flight job's current status
    unchanged. Never cached (see module docstring) - always re-runs
    fresh against a newly-discovered `domestic` dict, so this is
    "refreshed on demand via the button, or naturally on the next day"."""
    return _today_slot.start(_run_today)


def get_today_status() -> dict:
    return _today_slot.status()


# --- Volatility-indicator scan ---

_indicator_slot = _JobSlot()
_indicator_cache: dict = {"date": None, "threshold_pct": None, "result": None}
_indicator_discovery_cache: dict = _new_discovery_cache()


def _indicator_result(domestic: dict, threshold_pct: float, force: bool) -> pd.DataFrame:
    """Cached per (calendar day, threshold_pct) - see module docstring
    for why the threshold is part of the key. `force` bypasses the
    cache when `domestic` just changed (see _discover_and_filter_with_
    retry's own docstring for why)."""
    global _indicator_cache
    today = _today()
    if not force and _indicator_cache["date"] == today and _indicator_cache["threshold_pct"] == threshold_pct:
        logger.info("Reusing cached volatility-indicator result from %s (threshold=%s)", today, threshold_pct)
        return _indicator_cache["result"]
    result = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
    _indicator_cache = {"date": today, "threshold_pct": threshold_pct, "result": result}
    return result


def _run_indicator(slot: _JobSlot, started_at: str, threshold_pct: float) -> None:
    try:
        domestic, changed = _discover_and_filter_with_retry(_indicator_discovery_cache)
        failed_count = len(_indicator_discovery_cache["failed_symbols"])
        df = _indicator_result(domestic, threshold_pct, force=changed)
        slot.set_done(
            started_at, threshold_pct=threshold_pct, universe_size=len(domestic),
            failed_ticker_count=failed_count, volatility_indicator=_json_safe_records(df),
        )
        logger.info(
            "Volatility-indicator scan done: threshold=%s universe=%d matches=%d failed=%d",
            threshold_pct, len(domestic), len(df), failed_count,
        )
    except Exception as e:
        logger.exception("Volatility-indicator scan failed")
        slot.set_error(started_at, str(e), threshold_pct=threshold_pct)


def start_indicator_scan(threshold_pct: float) -> dict:
    """Starts a new volatility-indicator scan if none is currently
    running; otherwise returns the already-in-flight job's current status
    unchanged (including whatever threshold_pct that in-flight scan was
    started with - a second call with a different threshold while one is
    already running does NOT change or restart it, single-flight per
    scan TYPE, not per threshold - see module docstring)."""
    return _indicator_slot.start(_run_indicator, threshold_pct, threshold_pct=threshold_pct)


def get_indicator_status() -> dict:
    return _indicator_slot.status()

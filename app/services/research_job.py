"""
Background-job runner for the Swiss "today" (big-loss) volatility scan
(swiss_today_screener.py), backing app/routers/research.py's today
start/status endpoints. Rebound and volatility-indicator used to live
here too (three independent job slots) - moved 2026-08-19 to app/
services/scheduler.py's daily/monthly in-process cron instead, at the
user's explicit request to cut Yahoo Finance call volume for those two
tables (their underlying data doesn't change meaningfully more often than
that). "Today" stays here, unchanged: it's the one table where "as of
right now" is the whole point, so a cached/scheduled result would defeat
its purpose.

A full scan takes 1-3 minutes (universe discovery + ~100+ per-ticker
domicile checks + a batch price-history download), which is far past what
a single HTTP request should hold open - especially through
financial-sentiment-web's Vercel-hosted proxy, which has its own execution
time ceiling. So the scan runs in a plain background thread, detached from
the request that started it: POST .../start returns immediately with
"running", and the caller polls GET .../status until it sees "done" or
"error". This is NOT the asyncio.to_thread pattern used elsewhere in this
app (see fundamentals.py/earnings.py) - to_thread AWAITS completion within
the same request/response cycle, which is exactly what would recreate the
timeout problem this is meant to avoid.

Single global job slot: this app is a single Render worker, and running
two scans concurrently would just have them contend for the same yfinance
call budget for no benefit. If a scan is already running, starting a new
one just hands back the in-flight job's current status instead of
queuing, rejecting, or erroring - see _JobSlot.start.

No per-request/business-logic rate limiting at this layer (see
app/routers/research.py for why that was removed from the /start routes
2026-08-19) - the single-flight behavior above IS the protection against
wasted duplicate yfinance calls.

Module-level dict reassignment (`self.job = {...}`) inside _JobSlot is
used instead of mutating a shared dict in place - each state transition
is a single atomic pointer swap, so a status poll reading `self.job`
while the background thread "finishes" and reassigns it can't observe a
half-written state, without needing a lock around every read (CPython's
GIL makes the rebind itself atomic). The lock only guards the "is one
already running, if not start one" check in _JobSlot.start, which is the
one place two threads could otherwise race to both start a scan.

Universe: a single market-cap band (swiss_universe.MIN/MAX_MARKET_CAP_CHF,
CHF 500M+, no upper bound - that is the only requirement, including SMI
names) as of 2026-08-19.

Discovery caching + partial-failure retry (added 2026-08-19): also caches
its own `domestic` dict (and the `candidates` dict discovery produced it
from) per calendar day, via _discover_and_filter_with_retry below.
Confirmed live: some tickers' .info fetch can fail transiently (Yahoo
rate-limiting mid-scan) while most succeed - swiss_universe.filter_
domestic already failed soft per-ticker (drops just that one, doesn't
abort the scan), but the dropped ticker was gone for good even though the
next same-day click re-discovered the whole universe from scratch anyway,
paying the full ~150-call cost AGAIN just to end up with the same partial
result if Yahoo was still degraded, or a fresh full result that silently
discarded whatever succeeded the first time if only some tickers were
still failing. Now: a same-day re-scan reuses the cached `domestic` dict
outright and retries ONLY the specific tickers that failed last time
(their original discovery-time `quote` dict is kept in `candidates`
specifically so this retry doesn't need to re-run discovery itself) -
newly-succeeding tickers get merged permanently into the day's cached
`domestic`; still-failing ones stay in `failed_symbols` for the NEXT
same-day retry. `failed_ticker_count` is surfaced in the scan's own
"done"/"error" status so the frontend can show a warning when a result
may be incomplete rather than looking like a clean, complete scan.

No lock needed around this cache: _run_today only ever executes one at a
time (see _JobSlot's single-flight reasoning above), so there's no
concurrent writer to race against.

Rate-limit cooldown (added 2026-08-20): confirmed live - a genuine Yahoo
rate-limit block (YFRateLimitError, distinct from an ordinary per-ticker
fetch failure - see swiss_universe.filter_domestic's own comment) was
being treated exactly like any other transient failure by the discovery-
retry logic above: every "today" scan click blindly re-attempted the
full failed_symbols list, which just re-triggered the same block again -
the "running 2-3 scans throws this again" problem. _yahoo_rate_limited_
until below is a cooldown: while active, _discover_and_filter_with_retry
skips live calls entirely and returns whatever's already cached for
today (which may be empty, on a fresh day that got blocked before
anything was fetched - see that function's own comment for how that
case is surfaced to the frontend, rather than a raw YFRateLimitError
string)."""

import datetime
import logging
import threading

import pandas as pd

from app.services import swiss_today_screener, swiss_universe

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
    whatever extra fields it needs without this shared class needing to
    know their shape."""

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


# How long to stop attempting live filter_domestic calls after a
# confirmed Yahoo rate-limit hit - see module docstring's "Rate-limit
# cooldown" section. Not scientifically derived (no documented recovery
# window from Yahoo) - a conservative first guess, long enough that a
# user clicking Refresh a few times in a row can't keep re-triggering the
# block, short enough that a same-session retry later that day still has
# a real chance of working. Revisit if live experience shows it's too
# short (still hitting blocks right after cooldown expires) or too long
# (Yahoo was clearly fine again well before this elapsed).
RATE_LIMIT_COOLDOWN_MINUTES = 20

# None = no active cooldown.
_yahoo_rate_limited_until: datetime.datetime | None = None


def _yahoo_cooldown_remaining() -> datetime.datetime | None:
    """Returns the cooldown's expiry time if one is currently active,
    else None. Factored out so both the skip-live-calls check and the
    user-facing message below read the exact same value."""
    if _yahoo_rate_limited_until is not None and datetime.datetime.now(datetime.timezone.utc) < _yahoo_rate_limited_until:
        return _yahoo_rate_limited_until
    return None


def _start_yahoo_cooldown() -> None:
    global _yahoo_rate_limited_until
    _yahoo_rate_limited_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=RATE_LIMIT_COOLDOWN_MINUTES)
    logger.warning("Yahoo rate limit hit - suppressing further live discovery/filter calls until %s", _yahoo_rate_limited_until.isoformat())


class YahooRateLimitedError(Exception):
    """Raised only when a scan is rate-limited AND has nothing cached for
    today to fall back to (see _discover_and_filter_with_retry) - a
    clear, user-facing message for _JobSlot.set_error instead of a raw
    YFRateLimitError string, distinguishing "Yahoo is temporarily
    blocking this" from a genuine bug."""


def _rate_limit_message(until: datetime.datetime) -> str:
    return f"Yahoo Finance is currently rate-limiting these requests. Try again after {until.strftime('%H:%M UTC')}."


def _discover_and_filter_with_retry(cache: dict) -> tuple[dict, bool]:
    """Discovery+domicile-filter, cached per calendar day with same-day
    partial-failure retry (see module docstring for the full "why").
    `cache` is _today_discovery_cache - mutated in place. Returns
    (domestic, changed).

    `changed` is True whenever `domestic` is DIFFERENT from what it was
    on this cache's previous call (a fresh day's full discovery, or a
    same-day retry that actually recovered at least one previously-
    failed ticker). `changed` is False only for a genuine same-day no-op
    re-check (no failed_symbols left, or a retry that didn't recover
    anything new).

    Fresh day (or never run): full discover_candidates() + filter_
    domestic() pass, caches candidates/domestic/failed_symbols.

    Same day, with leftover failed_symbols from an earlier call today:
    retries ONLY those specific symbols (their original `quote` dict is
    still in `candidates`, so this doesn't need to re-run discovery) -
    newly-successful ones are merged into the cached `domestic`
    permanently for the rest of the day; still-failing ones remain in
    `failed_symbols` for the next call to retry again.

    Same day, no failed_symbols: pure cache hit, no yfinance calls at
    all.

    Rate-limit cooldown (see module docstring): checked FIRST, before any
    of the above - while active, this makes NO live calls at all. Serves
    today's cached `domestic` if there is one (even if it's incomplete
    from a partial fetch before the block hit - partial real data beats
    nothing), otherwise raises YahooRateLimitedError with a clear,
    actionable message rather than attempting a live call already known
    to fail. Whenever a live call below DOES hit a fresh rate limit, it
    starts the cooldown for next time - this function's own current call
    still tried live and reports whatever it got, cooldown only affects
    subsequent calls.
    """
    today = _today()

    cooldown_until = _yahoo_cooldown_remaining()
    if cooldown_until is not None:
        if cache["date"] == today and cache["domestic"]:
            logger.info(
                "Yahoo cooldown active until %s - reusing today's cached domestic (%d tickers) instead of a live call",
                cooldown_until.isoformat(), len(cache["domestic"]),
            )
            return cache["domestic"], False
        raise YahooRateLimitedError(_rate_limit_message(cooldown_until))

    if cache["date"] != today:
        candidates = swiss_universe.discover_candidates()
        domestic, failed_symbols, hit_rate_limit = swiss_universe.filter_domestic(candidates)
        cache["date"] = today
        cache["candidates"] = candidates
        cache["domestic"] = domestic
        cache["failed_symbols"] = failed_symbols
        logger.info(
            "Discovery for %s: universe=%d failed=%d", today, len(domestic), len(failed_symbols),
        )
        if hit_rate_limit:
            _start_yahoo_cooldown()
            if not domestic:
                raise YahooRateLimitedError(_rate_limit_message(_yahoo_rate_limited_until))
        return domestic, True

    if cache["failed_symbols"]:
        retry_candidates = {s: cache["candidates"][s] for s in cache["failed_symbols"] if s in cache["candidates"]}
        newly_domestic, still_failed, hit_rate_limit = swiss_universe.filter_domestic(retry_candidates)
        cache["failed_symbols"] = still_failed
        logger.info(
            "Retried %d previously-failed tickers: %d now succeeded, %d still failing",
            len(retry_candidates), len(newly_domestic), len(still_failed),
        )
        if hit_rate_limit:
            _start_yahoo_cooldown()
        if newly_domestic:
            cache["domestic"].update(newly_domestic)
            return cache["domestic"], True
        if hit_rate_limit and not cache["domestic"]:
            raise YahooRateLimitedError(_rate_limit_message(_yahoo_rate_limited_until))
    return cache["domestic"], False


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

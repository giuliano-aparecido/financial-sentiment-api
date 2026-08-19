"""
In-process cron for the two "cache the scan result, don't run it live"
research tables - rebound (daily) and volatility-indicator (monthly, all
three thresholds) - added at the user's explicit request to cut Yahoo
Finance calls: "table volatility... executed once a week or even month...
table rebound: it can be execute once a day." Deliberately NOT GitHub
Actions or any other external trigger (also explicit: "i dont want to rely
on github and it is business job, not infra like the keep warm script") -
APScheduler's AsyncIOScheduler runs entirely inside this process, started
from main.py's lifespan alongside inference.start_client(). The "today"
big-loss screener is untouched - it stays live/on-demand (see
research_job.py, app/routers/research.py), because "today" is only
meaningful as of right now, unlike these two.

Manual trigger, unified with the cron (added 2026-08-19 at the user's
explicit request: "It should be possible to refresh them manually as
before... While execution is going on, the button should be disabled...
also during cron job execution"): trigger_rebound_scan()/trigger_
indicator_scan() (called from app/routers/research.py's POST .../start)
run through the EXACT SAME guarded pipeline as the cron jobs below
(_run_rebound_scan_guarded/_run_indicator_scans_guarded), not a separate
code path - so is_rebound_scan_running()/is_indicator_scan_running()
(surfaced in the GET .../rebound and .../indicator read endpoints) is
accurate regardless of whether a scan in progress was started by the
schedule or by a user clicking Refresh. A single threading.Lock per table
guards both entry points against ever running two overlapping scans of
the same table, whatever the trigger source.

Reliability note: this process is on Render's free tier (15-minute
spin-down), currently kept warm 24/7 by an external UptimeRobot ping to
/health - unrelated to this feature, already true for the live /api/
analyze endpoint. That keeps APScheduler's cron firing close to on-time in
the common case, but isn't something this module depends on for
correctness: _catch_up_if_stale runs on every app startup and immediately
schedules a same-process run for whichever table hasn't been scanned
within its expected cadence, so a scan that got missed (a redeploy landing
exactly at 04:00, a rare UptimeRobot gap) just runs a bit late instead of
being silently skipped - fine for tables whose whole premise is "this
doesn't change day to day."

Batching (also explicit, "divide the swiss universe in 3, 4 or 5 batches
and execute the scan with waiting time between the batches"): layered on
TOP of swiss_universe.filter_domestic's existing per-ticker pacing/bounded
concurrency (INFO_REQUEST_DELAY_SECONDS/INFO_MAX_WORKERS - unchanged, see
that module), not a replacement for it - see filter_domestic_batched's own
docstring for why an unattended overnight job can afford to be far more
conservative than a live, user-triggered one.
"""

import datetime
import logging
import os
import threading

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services import scan_persistence, swiss_crash_rebound, swiss_universe, swiss_volatility_indicator
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS

logger = logging.getLogger(__name__)

# --- Config (env-overridable, sane defaults - early morning CET/CEST is
# well outside SIX Swiss Exchange trading hours, so the discovery/download
# calls this makes don't compete with live intraday activity either). ---

REBOUND_SCAN_HOUR = int(os.getenv("REBOUND_SCAN_HOUR", "6"))
REBOUND_SCAN_MINUTE = int(os.getenv("REBOUND_SCAN_MINUTE", "0"))

VOLATILITY_SCAN_DAY = int(os.getenv("VOLATILITY_SCAN_DAY", "1"))  # day of month, 1-28
VOLATILITY_SCAN_HOUR = int(os.getenv("VOLATILITY_SCAN_HOUR", "4"))
VOLATILITY_SCAN_MINUTE = int(os.getenv("VOLATILITY_SCAN_MINUTE", "0"))

NUM_SCAN_BATCHES = int(os.getenv("RESEARCH_SCAN_NUM_BATCHES", "4"))
SCAN_BATCH_DELAY_SECONDS = float(os.getenv("RESEARCH_SCAN_BATCH_DELAY_SECONDS", "60"))


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    # Same NaN->None handling as research_job.py's own helper (see that
    # module's docstring for why) - duplicated rather than imported since
    # that one is intentionally module-private and this module's job
    # pipeline is otherwise fully independent of research_job.py's
    # threaded single-flight machinery.
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def _discover_domestic_batched() -> tuple[dict, list[str]]:
    candidates = swiss_universe.discover_candidates()
    domestic, failed_symbols = swiss_universe.filter_domestic_batched(
        candidates, NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS,
    )
    if failed_symbols:
        logger.warning(
            "Scheduled scan discovery: %d/%d tickers failed (proceeding without them): %s",
            len(failed_symbols), len(candidates), failed_symbols,
        )
    return domestic, failed_symbols


# --- Rebound: guarded execution, shared by the cron job and the manual
# trigger (see module docstring) ---

_rebound_lock = threading.Lock()
_rebound_running = False


def is_rebound_scan_running() -> bool:
    return _rebound_running


def _run_rebound_scan() -> None:
    """The actual discovery + scan + persist work - real network calls,
    can take many minutes once batching delays are added. Never call this
    directly; go through _run_rebound_scan_guarded (via the cron job or
    trigger_rebound_scan below) so is_rebound_scan_running() stays
    accurate regardless of who started it."""
    logger.info("Rebound scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, _failed = _discover_domestic_batched()
    df = swiss_crash_rebound.run_scan(domestic)
    rows = _json_safe_records(df)
    scan_persistence.save_rebound_scan(rows)
    logger.info("Rebound scan done: universe=%d matches=%d", len(domestic), len(rows))


def _run_rebound_scan_guarded() -> None:
    """Registered as the cron job function AND run as the manual
    trigger's background-thread target (see trigger_rebound_scan) - the
    one place _rebound_running is set/cleared, so a manual click and a
    scheduled firing are indistinguishable to is_rebound_scan_running()/
    the frontend. Safe to register directly with AsyncIOScheduler as a
    plain sync function - its AsyncIOExecutor runs non-coroutine job
    functions via the event loop's default executor (a thread pool), not
    on the event loop itself, so this never blocks request handling
    while it runs (see apscheduler.executors.asyncio.AsyncIOExecutor)."""
    global _rebound_running
    with _rebound_lock:
        if _rebound_running:
            logger.info("Rebound scan requested but one is already running - no-op")
            return
        _rebound_running = True
    try:
        _run_rebound_scan()
    except Exception:
        logger.exception("Rebound scan failed")
    finally:
        with _rebound_lock:
            _rebound_running = False


def trigger_rebound_scan() -> bool:
    """Manual trigger (POST /api/research/volatility/rebound/start) -
    starts a background thread running the SAME guarded pipeline the
    cron uses. Returns True if this call started a new run, False if one
    was already in progress (single-flight, matching the button's old
    "clicking while already running is a safe no-op" behavior - see
    _run_rebound_scan_guarded's own lock for why this is authoritative
    even if two callers race here)."""
    if _rebound_running:
        return False
    threading.Thread(target=_run_rebound_scan_guarded, daemon=True).start()
    return True


# --- Volatility-indicator: same guarded-execution shape as rebound ---

_indicator_lock = threading.Lock()
_indicator_running = False


def is_indicator_scan_running() -> bool:
    return _indicator_running


def _run_indicator_scans() -> None:
    """Runs ALL of ALLOWED_THRESHOLD_PCTS off ONE shared discovery pass
    (not one discovery per threshold) - the frontend's threshold selector
    needs every value ready to read, and re-discovering the same
    ~150-ticker universe 3x for the same scan run would triple the Yahoo
    call count for zero benefit. Never call directly - see _run_rebound_
    scan's own docstring for why (identical reasoning)."""
    logger.info("Volatility-indicator scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, _failed = _discover_domestic_batched()
    scan_run_at = _now()
    for threshold_pct in ALLOWED_THRESHOLD_PCTS:
        df = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
        rows = _json_safe_records(df)
        scan_persistence.save_indicator_scan(rows, threshold_pct, scan_run_at)
        logger.info(
            "Volatility-indicator scan done: threshold=%s universe=%d matches=%d",
            threshold_pct, len(domestic), len(rows),
        )


def _run_indicator_scans_guarded() -> None:
    global _indicator_running
    with _indicator_lock:
        if _indicator_running:
            logger.info("Volatility-indicator scan requested but one is already running - no-op")
            return
        _indicator_running = True
    try:
        _run_indicator_scans()
    except Exception:
        logger.exception("Volatility-indicator scan failed")
    finally:
        with _indicator_lock:
            _indicator_running = False


def trigger_indicator_scan() -> bool:
    """Manual trigger (POST /api/research/volatility/indicator/start) -
    same shape as trigger_rebound_scan. Always scans every threshold (see
    _run_indicator_scans) regardless of which one the frontend currently
    has selected - there's no per-threshold "running" state, just one
    shared run covering all three."""
    if _indicator_running:
        return False
    threading.Thread(target=_run_indicator_scans_guarded, daemon=True).start()
    return True


def _is_rebound_stale() -> bool:
    _, last_run_at = scan_persistence.get_latest_rebound_scan()
    return last_run_at is None or last_run_at.date() < _now().date()


def _is_indicator_stale() -> bool:
    # Staleness is checked against ONE threshold (they're always written
    # together by _run_indicator_scans - see its own docstring), not all
    # three separately.
    now = _now()
    _, last_run_at = scan_persistence.get_latest_indicator_scan(ALLOWED_THRESHOLD_PCTS[0])
    return last_run_at is None or (last_run_at.year, last_run_at.month) != (now.year, now.month)


_scheduler: AsyncIOScheduler | None = None


def start() -> None:
    """Starts the in-process cron - called from main.py's lifespan.
    Idempotent-ish (a second call replaces the module-level scheduler
    reference), but main.py only ever calls this once per process.

    No-ops under RESEARCH_SCHEDULER_DISABLED=1 (set by conftest.py) -
    without this, every test using the `client` fixture enters the app's
    real lifespan (TestClient as a context manager), and the catch-up
    check below (_is_rebound_stale/_is_indicator_stale) would find the
    real Neon tables genuinely empty/stale on every run and schedule a
    REAL live Yahoo-scanning catch-up job - confirmed live: this hung the
    test suite past its 120s timeout the first time this module shipped
    without the guard.
    """
    if os.getenv("RESEARCH_SCHEDULER_DISABLED") == "1":
        logger.info("Research-scan scheduler disabled (RESEARCH_SCHEDULER_DISABLED=1)")
        return

    global _scheduler
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        _run_rebound_scan_guarded, CronTrigger(hour=REBOUND_SCAN_HOUR, minute=REBOUND_SCAN_MINUTE),
        id="rebound_scan", max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        _run_indicator_scans_guarded,
        CronTrigger(day=VOLATILITY_SCAN_DAY, hour=VOLATILITY_SCAN_HOUR, minute=VOLATILITY_SCAN_MINUTE),
        id="volatility_indicator_scan", max_instances=1, coalesce=True,
    )

    # Catch-up: schedule an immediate one-off run (not a direct blocking
    # call here) for any table that's stale, so app startup itself stays
    # fast - see module docstring for why this matters more than trusting
    # the cron triggers alone.
    if _is_rebound_stale():
        logger.info("Rebound scan catch-up: no scan for today yet, scheduling an immediate run")
        scheduler.add_job(_run_rebound_scan_guarded, id="rebound_scan_catchup", max_instances=1)
    if _is_indicator_stale():
        logger.info("Volatility-indicator scan catch-up: no scan this month yet, scheduling an immediate run")
        scheduler.add_job(_run_indicator_scans_guarded, id="volatility_indicator_scan_catchup", max_instances=1)

    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Research-scan scheduler started: rebound daily at %02d:%02d UTC, "
        "volatility-indicator monthly on day %d at %02d:%02d UTC",
        REBOUND_SCAN_HOUR, REBOUND_SCAN_MINUTE, VOLATILITY_SCAN_DAY, VOLATILITY_SCAN_HOUR, VOLATILITY_SCAN_MINUTE,
    )


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def is_running() -> bool:
    return _scheduler is not None

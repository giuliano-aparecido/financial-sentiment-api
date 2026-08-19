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


def run_rebound_scan_job() -> None:
    """Blocking - discovery + scan + persist (real network calls, and can
    take many minutes once batching delays are added). Safe to register
    directly with AsyncIOScheduler as a plain sync function - its
    AsyncIOExecutor runs non-coroutine job functions via the event loop's
    default executor (a thread pool), not on the event loop itself, so
    this never blocks request handling while it runs (see apscheduler.
    executors.asyncio.AsyncIOExecutor)."""
    logger.info("Scheduled rebound scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, _failed = _discover_domestic_batched()
    df = swiss_crash_rebound.run_scan(domestic)
    rows = _json_safe_records(df)
    scan_persistence.save_rebound_scan(rows)
    logger.info("Scheduled rebound scan done: universe=%d matches=%d", len(domestic), len(rows))


def run_indicator_scans_job() -> None:
    """Blocking, same non-event-loop-blocking guarantee as run_rebound_
    scan_job (see its own docstring). Runs
    ALL of ALLOWED_THRESHOLD_PCTS off ONE shared discovery pass (not one
    discovery per threshold) - the frontend's threshold selector needs
    every value ready to read with no live scan, and re-discovering the
    same ~150-ticker universe 3x for the same scan run would triple the
    Yahoo call count for zero benefit."""
    logger.info("Scheduled volatility-indicator scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, _failed = _discover_domestic_batched()
    scan_run_at = _now()
    for threshold_pct in ALLOWED_THRESHOLD_PCTS:
        df = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
        rows = _json_safe_records(df)
        scan_persistence.save_indicator_scan(rows, threshold_pct, scan_run_at)
        logger.info(
            "Scheduled volatility-indicator scan done: threshold=%s universe=%d matches=%d",
            threshold_pct, len(domestic), len(rows),
        )


def _is_rebound_stale() -> bool:
    _, last_run_at = scan_persistence.get_latest_rebound_scan()
    return last_run_at is None or last_run_at.date() < _now().date()


def _is_indicator_stale() -> bool:
    # Staleness is checked against ONE threshold (they're always written
    # together by run_indicator_scans_job - see its own docstring), not
    # all three separately.
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
        run_rebound_scan_job, CronTrigger(hour=REBOUND_SCAN_HOUR, minute=REBOUND_SCAN_MINUTE),
        id="rebound_scan", max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        run_indicator_scans_job,
        CronTrigger(day=VOLATILITY_SCAN_DAY, hour=VOLATILITY_SCAN_HOUR, minute=VOLATILITY_SCAN_MINUTE),
        id="volatility_indicator_scan", max_instances=1, coalesce=True,
    )

    # Catch-up: schedule an immediate one-off run (not a direct blocking
    # call here) for any table that's stale, so app startup itself stays
    # fast - see module docstring for why this matters more than trusting
    # the cron triggers alone.
    if _is_rebound_stale():
        logger.info("Rebound scan catch-up: no scan for today yet, scheduling an immediate run")
        scheduler.add_job(run_rebound_scan_job, id="rebound_scan_catchup", max_instances=1)
    if _is_indicator_stale():
        logger.info("Volatility-indicator scan catch-up: no scan this month yet, scheduling an immediate run")
        scheduler.add_job(run_indicator_scans_job, id="volatility_indicator_scan_catchup", max_instances=1)

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

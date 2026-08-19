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

Manual trigger is ALWAYS a full scan (2026-08-19, revised at the user's
explicit follow-up request: "i think is better to not touch the refresh
button, it should be a hard refresh anyway... In case of failed ticker
we should show a new button" instead of Refresh silently choosing
retry-vs-full on its own). trigger_rebound_retry()/trigger_indicator_
retry() are the SEPARATE failed-tickers-only path (POST .../retry,
frontend only renders that button when failed_ticker_count > 0) -
sharing the SAME lock as the full-scan trigger so the two can never run
concurrently for the same table, but never chosen automatically by
trigger_rebound_scan()/trigger_indicator_scan() itself.

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
    """The actual full discovery + scan + persist work - real network
    calls, can take many minutes once batching delays are added. Always a
    FULL fresh scan - what the cron/catch-up runs (via _run_rebound_scan_
    guarded) AND what the manual Refresh button runs (trigger_rebound_
    scan calls the same guarded function - see module docstring for why
    Refresh never picks retry-vs-full on its own). Never call directly."""
    logger.info("Rebound scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, failed = _discover_domestic_batched()
    df = swiss_crash_rebound.run_scan(domestic)
    rows = _json_safe_records(df)
    scan_persistence.save_rebound_scan(rows, failed_tickers=failed)
    logger.info("Rebound scan done: universe=%d matches=%d failed=%d", len(domestic), len(rows), len(failed))


def _retry_failed_rebound_tickers(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    """Re-fetches ONLY the specific tickers that failed on scan_run_at's
    run (their info fetch errored, not a real exclusion - see swiss_
    universe.filter_domestic's own docstring), recomputes just their scan
    rows, and merges them into the existing persisted rows under that
    SAME scan_run_at - not a full rescan. A fresh discover_candidates()
    call is still needed (to get each retried symbol's current screener
    `quote` dict, required by filter_domestic - see _fetch_domestic_
    entry), but that's one cheap screener call, not ~150 .info fetches.
    Tickers that still fail stay in the run's failed_tickers list for the
    next retry; ones missing from the fresh discovery entirely (delisted,
    renamed) are silently dropped from future retries rather than retried
    forever."""
    logger.info("Retrying %d failed rebound tickers from %s", len(failed_tickers), scan_run_at.isoformat())
    candidates = swiss_universe.discover_candidates()
    retry_candidates = {s: candidates[s] for s in failed_tickers if s in candidates}
    domestic, still_failed = swiss_universe.filter_domestic(retry_candidates)
    rows = []
    if domestic:
        df = swiss_crash_rebound.run_scan(domestic)
        rows = _json_safe_records(df)
        scan_persistence.merge_rebound_retry_rows(scan_run_at, rows)
    scan_persistence.update_rebound_run_failed_tickers(scan_run_at, still_failed)
    logger.info(
        "Rebound retry done: %d/%d recovered, %d still failing",
        len(domestic), len(failed_tickers), len(still_failed),
    )


def _run_rebound_scan_guarded() -> None:
    """Registered as the cron job function AND run as the full-scan
    manual trigger's background-thread target (see trigger_rebound_scan)
    - the one place _rebound_running is set/cleared for a full scan, so a
    manual Refresh click and a scheduled firing are indistinguishable to
    is_rebound_scan_running()/the frontend. _retry_failed_rebound_
    tickers_guarded below is the SEPARATE failed-tickers-only path,
    sharing this same lock. Safe to register directly with AsyncIOScheduler
    as a plain sync function - its AsyncIOExecutor runs non-coroutine job
    functions via the event loop's default executor (a thread pool), not
    on the event loop itself, so this never blocks request handling while
    it runs (see apscheduler.executors.asyncio.AsyncIOExecutor)."""
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
    """Manual Refresh trigger (POST /api/research/volatility/rebound/
    start) - ALWAYS a full scan, a hard refresh, regardless of any
    leftover failed tickers from a previous run (see module docstring for
    why this doesn't pick retry-vs-full on its own - trigger_rebound_
    retry below is the separate, explicit path for that). Starts a
    background thread running the same guarded pipeline the cron uses.
    Returns True if this call started a new run, False if one was already
    in progress (single-flight - see _run_rebound_scan_guarded's own lock
    for why this is authoritative even if two callers race here)."""
    if _rebound_running:
        return False
    threading.Thread(target=_run_rebound_scan_guarded, daemon=True).start()
    return True


def _retry_failed_rebound_tickers_guarded(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    """trigger_rebound_retry's background-thread target - same lock as
    _run_rebound_scan_guarded (a retry and a full scan of the same table
    must never run concurrently), but runs _retry_failed_rebound_tickers
    instead."""
    global _rebound_running
    with _rebound_lock:
        if _rebound_running:
            logger.info("Rebound retry requested but a scan is already running - no-op")
            return
        _rebound_running = True
    try:
        _retry_failed_rebound_tickers(scan_run_at, failed_tickers)
    except Exception:
        logger.exception("Rebound retry failed")
    finally:
        with _rebound_lock:
            _rebound_running = False


def trigger_rebound_retry() -> bool:
    """Manual "Retry Failed Tickers" trigger (POST /api/research/
    volatility/rebound/retry) - the frontend only renders this button
    when failed_ticker_count > 0, but this re-checks server-side rather
    than trusting that (the count could have changed between page load
    and click). Retries ONLY today's leftover failed tickers - never a
    full scan (see trigger_rebound_scan for that). Returns False (no-op,
    same single-flight contract as trigger_rebound_scan) if a scan is
    already running, OR if there's genuinely nothing to retry (no scan
    today, or today's scan has no failures)."""
    if _rebound_running:
        return False
    _rows, scan_run_at, failed_tickers = scan_persistence.get_latest_rebound_scan()
    if not failed_tickers or scan_run_at is None or scan_run_at.date() != _now().date():
        return False
    threading.Thread(
        target=_retry_failed_rebound_tickers_guarded, args=(scan_run_at, failed_tickers), daemon=True,
    ).start()
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
    call count for zero benefit. Always a FULL fresh scan - what the
    cron/catch-up runs AND what the manual Refresh button runs, same
    "no automatic retry-vs-full choice" as rebound - see _run_rebound_
    scan's own docstring. Never call directly."""
    logger.info("Volatility-indicator scan starting (batches=%d, delay=%ss)", NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS)
    domestic, failed = _discover_domestic_batched()
    scan_run_at = _now()
    for threshold_pct in ALLOWED_THRESHOLD_PCTS:
        df = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
        rows = _json_safe_records(df)
        scan_persistence.save_indicator_scan(rows, threshold_pct, failed_tickers=failed, scan_run_at=scan_run_at)
        logger.info(
            "Volatility-indicator scan done: threshold=%s universe=%d matches=%d failed=%d",
            threshold_pct, len(domestic), len(rows), len(failed),
        )


def _retry_failed_indicator_tickers(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    """Same shape as _retry_failed_rebound_tickers - a recovered ticker
    needs its row recomputed separately for EACH threshold (a company can
    qualify for a lower bar but not a higher one), all merged under the
    SAME scan_run_at, with ONE shared failed_tickers update after the
    per-threshold loop (see IndicatorScanRun's own docstring for why
    that's not threshold-scoped)."""
    logger.info("Retrying %d failed volatility-indicator tickers from %s", len(failed_tickers), scan_run_at.isoformat())
    candidates = swiss_universe.discover_candidates()
    retry_candidates = {s: candidates[s] for s in failed_tickers if s in candidates}
    domestic, still_failed = swiss_universe.filter_domestic(retry_candidates)
    if domestic:
        for threshold_pct in ALLOWED_THRESHOLD_PCTS:
            df = swiss_volatility_indicator.run_scan(domestic, threshold_pct)
            rows = _json_safe_records(df)
            scan_persistence.merge_indicator_retry_rows(scan_run_at, threshold_pct, rows)
    scan_persistence.update_indicator_run_failed_tickers(scan_run_at, still_failed)
    logger.info(
        "Volatility-indicator retry done: %d/%d recovered, %d still failing",
        len(domestic), len(failed_tickers), len(still_failed),
    )


def _run_indicator_scans_guarded() -> None:
    """Cron job function AND the full-scan manual trigger's background-
    thread target. See _run_rebound_scan_guarded's own docstring
    (identical reasoning, this is its indicator counterpart)."""
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
    """Manual Refresh trigger (POST /api/research/volatility/indicator/
    start) - ALWAYS a full scan covering every threshold, regardless of
    which one the frontend currently has selected or any leftover failed
    tickers (see trigger_indicator_retry for that separate path, and
    module docstring for why). There's no per-threshold "running" state,
    just one shared run covering all three."""
    if _indicator_running:
        return False
    threading.Thread(target=_run_indicator_scans_guarded, daemon=True).start()
    return True


def _retry_failed_indicator_tickers_guarded(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    """trigger_indicator_retry's background-thread target - same lock as
    _run_indicator_scans_guarded."""
    global _indicator_running
    with _indicator_lock:
        if _indicator_running:
            logger.info("Volatility-indicator retry requested but a scan is already running - no-op")
            return
        _indicator_running = True
    try:
        _retry_failed_indicator_tickers(scan_run_at, failed_tickers)
    except Exception:
        logger.exception("Volatility-indicator retry failed")
    finally:
        with _indicator_lock:
            _indicator_running = False


def trigger_indicator_retry() -> bool:
    """Manual "Retry Failed Tickers" trigger (POST /api/research/
    volatility/indicator/retry) - same shape as trigger_rebound_retry,
    scoped to calendar MONTH instead of day. Checked against ALLOWED_
    THRESHOLD_PCTS[0] only, same reasoning as _is_indicator_stale."""
    if _indicator_running:
        return False
    _rows, scan_run_at, failed_tickers = scan_persistence.get_latest_indicator_scan(ALLOWED_THRESHOLD_PCTS[0])
    now = _now()
    if not failed_tickers or scan_run_at is None or (scan_run_at.year, scan_run_at.month) != (now.year, now.month):
        return False
    threading.Thread(
        target=_retry_failed_indicator_tickers_guarded, args=(scan_run_at, failed_tickers), daemon=True,
    ).start()
    return True


def _is_rebound_stale() -> bool:
    _, last_run_at, _failed = scan_persistence.get_latest_rebound_scan()
    return last_run_at is None or last_run_at.date() < _now().date()


def _is_indicator_stale() -> bool:
    # Staleness is checked against ONE threshold (they're always written
    # together by _run_indicator_scans - see its own docstring), not all
    # three separately.
    now = _now()
    _, last_run_at, _failed = scan_persistence.get_latest_indicator_scan(ALLOWED_THRESHOLD_PCTS[0])
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

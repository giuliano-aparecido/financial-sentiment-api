"""
On-demand, single-flight runners for the two persisted research tables -
rebound and volatility-indicator (all three thresholds) - whose results
are cached in Postgres (scan_persistence.py) and served by the GET
.../rebound and .../indicator read endpoints until the user clicks
Refresh. Nothing in here runs on a schedule any more: the daily/monthly
APScheduler cron and its startup catch-up were removed (2026-09-17) so
that every Yahoo Finance request this app makes on its own initiative
goes to the twice-daily "today" big-loss alert (alerts.py); with the
alert waking the process twice a day, the old startup catch-up would have
launched a full rebound scan at exactly the same moment. The module keeps
its historical name.

trigger_rebound_scan()/trigger_indicator_scan() (POST .../start) run the
guarded pipeline (_run_rebound_scan_guarded/_run_indicator_scans_guarded)
in a background thread; is_rebound_scan_running()/is_indicator_scan_
running() (surfaced by the GET read endpoints) is the one source of truth
the frontend uses to disable the button. A single threading.Lock per
table guards against two overlapping scans of the same table.

A manual trigger is ALWAYS a full scan (2026-08-19, at the user's explicit
request: "i think is better to not touch the refresh button, it should be
a hard refresh anyway... In case of failed ticker we should show a new
button"). trigger_rebound_retry()/trigger_indicator_retry() are the
SEPARATE failed-tickers-only path (POST .../retry; the frontend only
renders that button when failed_ticker_count > 0), sharing the SAME lock
as the full-scan trigger so the two can never run concurrently for the
same table, but never chosen automatically by the full-scan trigger.

Batching (also explicit, "divide the swiss universe in 3, 4 or 5 batches
and execute the scan with waiting time between the batches"): layered on
TOP of swiss_universe.filter_domestic's existing per-ticker pacing/bounded
concurrency (INFO_REQUEST_DELAY_SECONDS/INFO_MAX_WORKERS - unchanged, see
that module), not a replacement for it. Kept for the on-demand runs too:
the user waits a few extra minutes behind a disabled button, but the
Yahoo budget is the scarcer resource.
"""

import datetime
import logging
import os
import threading
from collections.abc import Callable

import pandas as pd

from app.services import scan_persistence, swiss_crash_rebound, swiss_universe, swiss_volatility_indicator
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS

logger = logging.getLogger(__name__)

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


def _run_guarded(
    lock: threading.Lock,
    get_running: Callable[[], bool],
    set_running: Callable[[bool], None],
    work: Callable[[], None],
    *,
    busy_message: str,
    failure_message: str,
) -> None:
    """Single-flight wrapper shared by the rebound/indicator scan+retry
    guarded entry points below: acquire lock, no-op if already running,
    else mark running, run `work` (logging+swallowing any exception so a
    failed scan doesn't crash the scheduler thread), then always clear
    the running flag. `get_running`/`set_running` read/write the
    relevant module-level `_rebound_running`/`_indicator_running` flag -
    passed as callables rather than a shared mutable object since these
    are plain module globals, not instance state."""
    with lock:
        if get_running():
            logger.info(busy_message)
            return
        set_running(True)
    try:
        work()
    except Exception:
        logger.exception(failure_message)
    finally:
        with lock:
            set_running(False)


def _discover_domestic_batched() -> tuple[dict, list[str]]:
    candidates = swiss_universe.discover_candidates()
    # `hit_rate_limit` (2026-08-20 addition to filter_domestic_batched -
    # see that function's own docstring) isn't consumed here: this path
    # is already the most rate-limit-conscious one in the app (batched,
    # minutes of delay between chunks), and a hit here just means
    # whatever candidates weren't reached land in failed_symbols same as
    # any other failure, picked up by the next manual scan or a
    # manual retry - no separate cooldown gate needed on top of that.
    domestic, failed_symbols, _hit_rate_limit = swiss_universe.filter_domestic_batched(
        candidates, NUM_SCAN_BATCHES, SCAN_BATCH_DELAY_SECONDS,
    )
    if failed_symbols:
        logger.warning(
            "Scan discovery: %d/%d tickers failed (proceeding without them): %s",
            len(failed_symbols), len(candidates), failed_symbols,
        )
    return domestic, failed_symbols


# --- Rebound: guarded execution (see module docstring) ---

_rebound_lock = threading.Lock()
_rebound_running = False


def is_rebound_scan_running() -> bool:
    return _rebound_running


def _set_rebound_running(value: bool) -> None:
    global _rebound_running
    _rebound_running = value


def _run_rebound_scan() -> None:
    """The actual full discovery + scan + persist work - real network
    calls, can take many minutes with the batching delays. Always a FULL
    fresh scan (see module docstring for why Refresh never picks
    retry-vs-full on its own). Never call directly - go through
    _run_rebound_scan_guarded."""
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
    domestic, still_failed, _hit_rate_limit = swiss_universe.filter_domestic(retry_candidates)
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
    """The full-scan trigger's background-thread target (see
    trigger_rebound_scan) - the one place _rebound_running is set/cleared
    for a full scan. _retry_failed_rebound_tickers_guarded below is the
    SEPARATE failed-tickers-only path, sharing this same lock."""
    _run_guarded(
        _rebound_lock, is_rebound_scan_running, _set_rebound_running, _run_rebound_scan,
        busy_message="Rebound scan requested but one is already running - no-op",
        failure_message="Rebound scan failed",
    )


def trigger_rebound_scan() -> bool:
    """Manual Refresh trigger (POST /api/research/volatility/rebound/
    start) - ALWAYS a full scan, a hard refresh, regardless of any
    leftover failed tickers from a previous run (see module docstring for
    why this doesn't pick retry-vs-full on its own - trigger_rebound_
    retry below is the separate, explicit path for that). Starts a
    background thread running the guarded pipeline.
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
    _run_guarded(
        _rebound_lock, is_rebound_scan_running, _set_rebound_running,
        lambda: _retry_failed_rebound_tickers(scan_run_at, failed_tickers),
        busy_message="Rebound retry requested but a scan is already running - no-op",
        failure_message="Rebound retry failed",
    )


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


def _set_indicator_running(value: bool) -> None:
    global _indicator_running
    _indicator_running = value


def _run_indicator_scans() -> None:
    """Runs ALL of ALLOWED_THRESHOLD_PCTS off ONE shared discovery pass
    (not one discovery per threshold) - the frontend's threshold selector
    needs every value ready to read, and re-discovering the same
    ~150-ticker universe 3x for the same scan run would triple the Yahoo
    call count for zero benefit. Always a FULL fresh scan, same "no
    automatic retry-vs-full choice" as rebound - see _run_rebound_scan's
    own docstring. Never call directly."""
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
    domestic, still_failed, _hit_rate_limit = swiss_universe.filter_domestic(retry_candidates)
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
    """The full-scan trigger's background-thread target - see
    _run_rebound_scan_guarded (this is its indicator counterpart)."""
    _run_guarded(
        _indicator_lock, is_indicator_scan_running, _set_indicator_running, _run_indicator_scans,
        busy_message="Volatility-indicator scan requested but one is already running - no-op",
        failure_message="Volatility-indicator scan failed",
    )


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
    _run_guarded(
        _indicator_lock, is_indicator_scan_running, _set_indicator_running,
        lambda: _retry_failed_indicator_tickers(scan_run_at, failed_tickers),
        busy_message="Volatility-indicator retry requested but a scan is already running - no-op",
        failure_message="Volatility-indicator retry failed",
    )


def trigger_indicator_retry() -> bool:
    """Manual "Retry Failed Tickers" trigger (POST /api/research/
    volatility/indicator/retry) - same shape as trigger_rebound_retry,
    scoped to calendar MONTH instead of day. Checked against ALLOWED_
    THRESHOLD_PCTS[0] only - all three thresholds share one scan_run_at
    (see _run_indicator_scans)."""
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

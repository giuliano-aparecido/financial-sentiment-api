"""
Persistence layer backing the on-demand research scans (see
app/services/scheduler.py): writes one scan run's rows to Neon Postgres,
and reads back the latest run's rows for the read-only endpoints in
app/routers/research.py. Deliberately dumb - no business logic here, just
save/load - so scheduler.py stays the one place that runs a scan (full or
failed-ticker-only retry, both user-triggered) and
research.py the one place that decides how it's exposed over HTTP.

Replace-on-write, not append-forever: each save_* call first deletes every
row with an OLDER scan_run_at than the one being written (see the
delete(...).where(... < scan_run_at) calls below) rather than keeping
every historical run. Keeping full history was considered (cheap at this
row count, free trend data later) but rejected for now - nothing in this
feature reads anything but the latest run, and an unbounded table growing
by one scan's worth of rows on every Refresh forever is a maintenance
question nobody's asked for yet. If history ever becomes wanted, this is
the one place to change.

Failed-ticker tracking (ReboundScanRun/IndicatorScanRun - added
2026-08-19 at the user's explicit request): a scan run persists not just
its successfully-fetched company rows but ALSO the list of tickers whose
.info fetch failed (see swiss_universe.filter_domestic's own docstring on
failed_symbols) - a real, temporary fetch failure, not a company that was
correctly excluded. This is what lets the frontend show "N companies
missing, failed to fetch" instead of silently rendering an incomplete
table as if it were complete, and lets a manual Refresh retry ONLY those
specific tickers (merge_*_retry_rows/update_*_run_failed_tickers below)
instead of redoing the whole scan. Kept in Neon (not in-memory, unlike
the OLDER research_job.py's same-day discovery-retry cache for "today")
because a retry needs to work correctly even if Render restarted between
the original scan and the retry.
"""

import datetime
import logging

from sqlalchemy import delete, func, select, update

from app.db.models import IndicatorScanRun, ReboundScanRun, ReboundScanRow, VolatilityIndicatorScanRow
from app.db.session import get_session

logger = logging.getLogger(__name__)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# --- Rebound ---


def save_rebound_scan(
    rows: list[dict], failed_tickers: list[str] | None = None, scan_run_at: datetime.datetime | None = None,
) -> None:
    scan_run_at = scan_run_at or _now()
    failed_tickers = failed_tickers or []
    with get_session() as session:
        session.execute(delete(ReboundScanRow).where(ReboundScanRow.scan_run_at < scan_run_at))
        session.execute(delete(ReboundScanRun).where(ReboundScanRun.scan_run_at < scan_run_at))
        session.add_all([
            ReboundScanRow(scan_run_at=scan_run_at, ticker=row["ticker"], data=row)
            for row in rows
        ])
        session.add(ReboundScanRun(scan_run_at=scan_run_at, failed_tickers=failed_tickers))
        session.commit()
    logger.info(
        "Saved rebound scan: %d rows, %d failed tickers, at %s",
        len(rows), len(failed_tickers), scan_run_at.isoformat(),
    )


def get_latest_rebound_scan() -> tuple[list[dict], datetime.datetime | None, list[str]]:
    """Returns (rows, scan_run_at, failed_tickers) for the most recent
    saved scan - ([], None, []) if nothing has ever been saved (no one
    has clicked Refresh yet)."""
    with get_session() as session:
        latest = session.execute(select(func.max(ReboundScanRow.scan_run_at))).scalar_one_or_none()
        if latest is None:
            return [], None, []
        rows = session.execute(
            select(ReboundScanRow.data).where(ReboundScanRow.scan_run_at == latest)
        ).scalars().all()
        failed_tickers = session.execute(
            select(ReboundScanRun.failed_tickers).where(ReboundScanRun.scan_run_at == latest)
        ).scalar_one_or_none() or []
        return list(rows), latest, list(failed_tickers)


def merge_rebound_retry_rows(scan_run_at: datetime.datetime, rows: list[dict]) -> None:
    """Upserts newly-recovered rows into an EXISTING scan_run_at's rows -
    per-ticker delete-then-insert (mirroring save_rebound_scan's own
    replace-on-write pattern), not a bare append. Called by
    scheduler._retry_failed_rebound_tickers once per retry; pair with
    update_rebound_run_failed_tickers to record which tickers, if any,
    are still failing after the retry.

    The per-ticker delete matters because this call and
    update_rebound_run_failed_tickers commit in SEPARATE sessions/
    transactions (see scheduler._retry_failed_rebound_tickers): a crash
    in between leaves a ticker both persisted here AND still listed as
    failed, so the NEXT retry re-fetches and re-merges that SAME ticker.
    A plain append would then insert a second row for the same
    ticker/scan_run_at instead of replacing the first - deleting any
    existing row for these tickers first keeps a retry-after-crash
    idempotent."""
    if not rows:
        return
    tickers = [row["ticker"] for row in rows]
    with get_session() as session:
        session.execute(
            delete(ReboundScanRow).where(
                ReboundScanRow.scan_run_at == scan_run_at,
                ReboundScanRow.ticker.in_(tickers),
            )
        )
        session.add_all([
            ReboundScanRow(scan_run_at=scan_run_at, ticker=row["ticker"], data=row)
            for row in rows
        ])
        session.commit()


def update_rebound_run_failed_tickers(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    with get_session() as session:
        session.execute(
            update(ReboundScanRun).where(ReboundScanRun.scan_run_at == scan_run_at).values(failed_tickers=failed_tickers)
        )
        session.commit()


# --- Volatility indicator ---


def save_indicator_scan(
    rows: list[dict], threshold_pct: float, failed_tickers: list[str] | None = None,
    scan_run_at: datetime.datetime | None = None,
) -> None:
    scan_run_at = scan_run_at or _now()
    failed_tickers = failed_tickers or []
    with get_session() as session:
        session.execute(
            delete(VolatilityIndicatorScanRow).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
                VolatilityIndicatorScanRow.scan_run_at < scan_run_at,
            )
        )
        session.execute(delete(IndicatorScanRun).where(IndicatorScanRun.scan_run_at < scan_run_at))
        session.add_all([
            VolatilityIndicatorScanRow(
                scan_run_at=scan_run_at, threshold_pct=threshold_pct, ticker=row["ticker"], data=row,
            )
            for row in rows
        ])
        # One shared run-meta row across all three thresholds (see
        # IndicatorScanRun's own docstring) - only insert it once, the
        # first threshold in this scan's save loop to hit this code path;
        # later thresholds in the SAME run share the same scan_run_at, so
        # skip re-inserting (would violate the unique constraint).
        existing = session.execute(
            select(IndicatorScanRun).where(IndicatorScanRun.scan_run_at == scan_run_at)
        ).scalar_one_or_none()
        if existing is None:
            session.add(IndicatorScanRun(scan_run_at=scan_run_at, failed_tickers=failed_tickers))
        session.commit()
    logger.info(
        "Saved volatility-indicator scan: threshold=%s %d rows, %d failed tickers, at %s",
        threshold_pct, len(rows), len(failed_tickers), scan_run_at.isoformat(),
    )


def get_latest_indicator_scan(threshold_pct: float) -> tuple[list[dict], datetime.datetime | None, list[str]]:
    """Same as get_latest_rebound_scan, scoped to one threshold_pct for
    the rows - `failed_tickers` is NOT threshold-scoped (see
    IndicatorScanRun's own docstring), it's the one shared list for
    whichever scan_run_at this threshold's own latest rows came from."""
    with get_session() as session:
        latest = session.execute(
            select(func.max(VolatilityIndicatorScanRow.scan_run_at)).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
            )
        ).scalar_one_or_none()
        if latest is None:
            return [], None, []
        rows = session.execute(
            select(VolatilityIndicatorScanRow.data).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
                VolatilityIndicatorScanRow.scan_run_at == latest,
            )
        ).scalars().all()
        failed_tickers = session.execute(
            select(IndicatorScanRun.failed_tickers).where(IndicatorScanRun.scan_run_at == latest)
        ).scalar_one_or_none() or []
        return list(rows), latest, list(failed_tickers)


def merge_indicator_retry_rows(scan_run_at: datetime.datetime, threshold_pct: float, rows: list[dict]) -> None:
    """Same upsert (delete-then-insert per ticker) as merge_rebound_retry_
    rows, scoped to one threshold_pct - see that function's own docstring
    for why the delete matters (a crash between this commit and
    update_indicator_run_failed_tickers's, across the per-threshold loop
    in scheduler._retry_failed_indicator_tickers, must not duplicate a
    ticker's row on the next retry). scheduler._retry_failed_indicator_
    tickers calls this once per threshold (a recovered ticker's row
    differs per threshold), then update_indicator_run_failed_tickers ONCE
    after the loop."""
    if not rows:
        return
    tickers = [row["ticker"] for row in rows]
    with get_session() as session:
        session.execute(
            delete(VolatilityIndicatorScanRow).where(
                VolatilityIndicatorScanRow.scan_run_at == scan_run_at,
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
                VolatilityIndicatorScanRow.ticker.in_(tickers),
            )
        )
        session.add_all([
            VolatilityIndicatorScanRow(
                scan_run_at=scan_run_at, threshold_pct=threshold_pct, ticker=row["ticker"], data=row,
            )
            for row in rows
        ])
        session.commit()


def update_indicator_run_failed_tickers(scan_run_at: datetime.datetime, failed_tickers: list[str]) -> None:
    with get_session() as session:
        session.execute(
            update(IndicatorScanRun).where(IndicatorScanRun.scan_run_at == scan_run_at).values(failed_tickers=failed_tickers)
        )
        session.commit()

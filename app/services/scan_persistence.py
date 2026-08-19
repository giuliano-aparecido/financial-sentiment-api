"""
Persistence layer backing the scheduled research scans (see
app/services/scheduler.py): writes one scan run's rows to Neon Postgres,
and reads back the latest run's rows for the read-only endpoints in
app/routers/research.py. Deliberately dumb - no business logic here, just
save/load - so scheduler.py stays the one place that decides WHEN a scan
runs and research.py the one place that decides how it's exposed over
HTTP.

Replace-on-write, not append-forever: each save_* call first deletes every
row with an OLDER scan_run_at than the one being written (see _prune_
older_than) rather than keeping every historical run. Keeping full history
was considered (cheap at this row count, free trend data later) but
rejected for now - nothing in this feature reads anything but the latest
run, and an unbounded table growing by one scan's worth of rows every
day/month forever is a maintenance question nobody's asked for yet. If
history ever becomes wanted, this is the one place to change.
"""

import datetime
import logging

from sqlalchemy import delete, func, select

from app.db.models import ReboundScanRow, VolatilityIndicatorScanRow
from app.db.session import get_session

logger = logging.getLogger(__name__)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def save_rebound_scan(rows: list[dict], scan_run_at: datetime.datetime | None = None) -> None:
    scan_run_at = scan_run_at or _now()
    with get_session() as session:
        session.execute(delete(ReboundScanRow).where(ReboundScanRow.scan_run_at < scan_run_at))
        session.add_all([
            ReboundScanRow(scan_run_at=scan_run_at, ticker=row["ticker"], data=row)
            for row in rows
        ])
        session.commit()
    logger.info("Saved rebound scan: %d rows at %s", len(rows), scan_run_at.isoformat())


def get_latest_rebound_scan() -> tuple[list[dict], datetime.datetime | None]:
    """Returns (rows, scan_run_at) for the most recent saved scan -
    ([], None) if nothing has ever been saved (e.g. the first scheduled
    run hasn't fired yet)."""
    with get_session() as session:
        latest = session.execute(select(func.max(ReboundScanRow.scan_run_at))).scalar_one_or_none()
        if latest is None:
            return [], None
        rows = session.execute(
            select(ReboundScanRow.data).where(ReboundScanRow.scan_run_at == latest)
        ).scalars().all()
        return list(rows), latest


def save_indicator_scan(
    rows: list[dict], threshold_pct: float, scan_run_at: datetime.datetime | None = None,
) -> None:
    scan_run_at = scan_run_at or _now()
    with get_session() as session:
        session.execute(
            delete(VolatilityIndicatorScanRow).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
                VolatilityIndicatorScanRow.scan_run_at < scan_run_at,
            )
        )
        session.add_all([
            VolatilityIndicatorScanRow(
                scan_run_at=scan_run_at, threshold_pct=threshold_pct, ticker=row["ticker"], data=row,
            )
            for row in rows
        ])
        session.commit()
    logger.info(
        "Saved volatility-indicator scan: threshold=%s %d rows at %s",
        threshold_pct, len(rows), scan_run_at.isoformat(),
    )


def get_latest_indicator_scan(threshold_pct: float) -> tuple[list[dict], datetime.datetime | None]:
    """Same as get_latest_rebound_scan, scoped to one threshold_pct - each
    threshold is scanned and stored independently (see scheduler.py), so
    "latest" is per-threshold, not global across all three."""
    with get_session() as session:
        latest = session.execute(
            select(func.max(VolatilityIndicatorScanRow.scan_run_at)).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
            )
        ).scalar_one_or_none()
        if latest is None:
            return [], None
        rows = session.execute(
            select(VolatilityIndicatorScanRow.data).where(
                VolatilityIndicatorScanRow.threshold_pct == threshold_pct,
                VolatilityIndicatorScanRow.scan_run_at == latest,
            )
        ).scalars().all()
        return list(rows), latest

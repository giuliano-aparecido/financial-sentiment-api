from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# JSONB on Postgres (Neon, production/dev), plain JSON everywhere else -
# `with_variant` swaps the storage type per-dialect without changing the
# Postgres schema Alembic generates (JSON().with_variant(JSONB(), "postgresql")
# renders identically to a bare JSONB column there). This is what lets
# tests use an in-memory SQLite engine (see tests/test_scan_persistence.py)
# instead of needing a real Postgres connection, or - worse - writing test
# rows into the real Neon tables where they could get served to a real
# user as "the latest scan" (save_* deletes older rows on every write).
_JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class ReboundScanRow(Base):
    """One crash-then-rebound match from one persisted scan run (see
    app/services/scheduler.py) - `data` is the full row dict exactly as
    swiss_crash_rebound.run_scan() produced it (ticker/name/sector/
    market_cap/loss_*/gain_*/... - see that function's own docstring for
    the full field list), stored as JSONB rather than one typed column per
    field so this table doesn't need a migration every time that scan
    output's shape changes. `ticker`/`scan_run_at` are pulled out as real
    columns because callers filter/sort on them directly (latest run's
    rows, one ticker's history) - duplicating them out of `data` is cheap
    and avoids a JSONB path query for the common case.
    """

    __tablename__ = "rebound_scan_rows"
    __table_args__ = (Index("ix_rebound_scan_rows_scan_run_at", "scan_run_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[dict] = mapped_column(_JSON_TYPE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ReboundScanRun(Base):
    """One row per rebound scan RUN (not per company - see ReboundScanRow
    for that), tracking which tickers failed to fetch during that run's
    discovery step and are therefore MISSING from its ReboundScanRow set -
    not because they were correctly excluded (wrong domicile, too
    illiquid), but because the .info fetch itself errored (see
    swiss_universe.filter_domestic's own docstring on failed_symbols).
    Lets the frontend show a "N companies missing, failed to fetch"
    warning, and lets a manual Refresh retry ONLY these specific tickers
    (see scheduler.py's _retry_failed_rebound_tickers) instead of
    redoing the whole ~150-ticker scan just to recover a handful."""

    __tablename__ = "rebound_scan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, unique=True)
    failed_tickers: Mapped[list] = mapped_column(_JSON_TYPE, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class IndicatorScanRun(Base):
    """Same purpose as ReboundScanRun, for the volatility-indicator scan.
    Not scoped by threshold_pct - one shared discovery pass covers all of
    swiss_volatility_indicator.ALLOWED_THRESHOLD_PCTS in a single run (see
    scheduler._run_indicator_scans's own docstring), so failed_tickers is
    the same regardless of which threshold is being viewed."""

    __tablename__ = "indicator_scan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, unique=True)
    failed_tickers: Mapped[list] = mapped_column(_JSON_TYPE, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class VolatilityIndicatorScanRow(Base):
    """One company's row from one persisted volatility-indicator scan run,
    for one threshold_pct (see swiss_volatility_indicator.ALLOWED_
    THRESHOLD_PCTS - one scan runs all three and stores each
    separately, so the frontend's threshold selector still works without
    a live scan). Same JSONB-blob-plus-a-few-real-columns shape as
    ReboundScanRow above, for the same reason."""

    __tablename__ = "volatility_indicator_scan_rows"
    __table_args__ = (
        Index("ix_volatility_indicator_scan_rows_scan_run_at_threshold", "scan_run_at", "threshold_pct"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    threshold_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[dict] = mapped_column(_JSON_TYPE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

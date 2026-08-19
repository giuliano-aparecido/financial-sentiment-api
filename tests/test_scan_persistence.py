import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db.models  # noqa: F401 - populates Base.metadata with both tables
from app.db.base import Base
from app.services import scan_persistence


@pytest.fixture
def sqlite_session(monkeypatch):
    # In-memory SQLite, not the real Neon DB - scan_persistence.save_*
    # deletes older rows on every write, so running these against the real
    # tables would risk a test row becoming "the latest scan" served to a
    # real user. See app/db/models.py's _JSON_TYPE comment for how the
    # JSONB column stays SQLite-compatible for exactly this reason.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    monkeypatch.setattr(scan_persistence, "get_session", lambda: session_local())
    return session_local


def _dt(y, mo, d, h=0, mi=0):
    return datetime.datetime(y, mo, d, h, mi)  # naive - see fixture's own note


# --- rebound ---


def test_get_latest_rebound_scan_returns_empty_when_nothing_saved(sqlite_session):
    rows, scan_run_at = scan_persistence.get_latest_rebound_scan()
    assert rows == []
    assert scan_run_at is None


def test_save_and_get_latest_rebound_scan(sqlite_session):
    rows = [{"ticker": "NESN.SW", "drop_pct": -5.2}, {"ticker": "ABBN.SW", "drop_pct": -6.1}]
    scan_persistence.save_rebound_scan(rows, scan_run_at=_dt(2026, 8, 19, 6, 0))

    got_rows, scan_run_at = scan_persistence.get_latest_rebound_scan()
    assert {r["ticker"] for r in got_rows} == {"NESN.SW", "ABBN.SW"}
    assert scan_run_at == _dt(2026, 8, 19, 6, 0)


def test_save_rebound_scan_replaces_older_rows(sqlite_session):
    scan_persistence.save_rebound_scan([{"ticker": "OLD.SW"}], scan_run_at=_dt(2026, 8, 18, 6, 0))
    scan_persistence.save_rebound_scan([{"ticker": "NEW.SW"}], scan_run_at=_dt(2026, 8, 19, 6, 0))

    rows, scan_run_at = scan_persistence.get_latest_rebound_scan()
    assert [r["ticker"] for r in rows] == ["NEW.SW"]
    assert scan_run_at == _dt(2026, 8, 19, 6, 0)


def test_save_rebound_scan_defaults_scan_run_at_to_now(sqlite_session):
    # SQLite's DateTime(timezone=True) doesn't reliably round-trip tzinfo
    # on read-back (a test-harness artifact of the in-memory SQLite engine
    # this fixture uses, not a real Postgres/Neon behavior) - comparing
    # naively here sidesteps that instead of asserting against it.
    before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    scan_persistence.save_rebound_scan([{"ticker": "AAPL"}])
    after = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    _rows, scan_run_at = scan_persistence.get_latest_rebound_scan()
    assert before <= scan_run_at.replace(tzinfo=None) <= after


# --- volatility indicator ---


def test_get_latest_indicator_scan_returns_empty_for_unscanned_threshold(sqlite_session):
    rows, scan_run_at = scan_persistence.get_latest_indicator_scan(3.0)
    assert rows == []
    assert scan_run_at is None


def test_save_and_get_latest_indicator_scan_scoped_per_threshold(sqlite_session):
    scan_persistence.save_indicator_scan([{"ticker": "A.SW"}], threshold_pct=2.0, scan_run_at=_dt(2026, 8, 1, 4, 0))
    scan_persistence.save_indicator_scan([{"ticker": "B.SW"}], threshold_pct=5.0, scan_run_at=_dt(2026, 8, 1, 4, 0))

    rows_2, run_at_2 = scan_persistence.get_latest_indicator_scan(2.0)
    rows_5, run_at_5 = scan_persistence.get_latest_indicator_scan(5.0)
    assert [r["ticker"] for r in rows_2] == ["A.SW"]
    assert [r["ticker"] for r in rows_5] == ["B.SW"]
    assert run_at_2 == run_at_5 == _dt(2026, 8, 1, 4, 0)


def test_save_indicator_scan_replaces_older_rows_for_same_threshold_only(sqlite_session):
    scan_persistence.save_indicator_scan([{"ticker": "OLD.SW"}], threshold_pct=2.0, scan_run_at=_dt(2026, 7, 1, 4, 0))
    scan_persistence.save_indicator_scan([{"ticker": "KEEP.SW"}], threshold_pct=5.0, scan_run_at=_dt(2026, 7, 1, 4, 0))
    scan_persistence.save_indicator_scan([{"ticker": "NEW.SW"}], threshold_pct=2.0, scan_run_at=_dt(2026, 8, 1, 4, 0))

    rows_2, _ = scan_persistence.get_latest_indicator_scan(2.0)
    rows_5, _ = scan_persistence.get_latest_indicator_scan(5.0)
    assert [r["ticker"] for r in rows_2] == ["NEW.SW"]
    assert [r["ticker"] for r in rows_5] == ["KEEP.SW"]  # untouched - different threshold

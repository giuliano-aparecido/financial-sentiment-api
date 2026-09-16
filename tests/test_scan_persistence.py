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
    rows, scan_run_at, failed = scan_persistence.get_latest_rebound_scan()
    assert rows == []
    assert scan_run_at is None
    assert failed == []


def test_save_and_get_latest_rebound_scan(sqlite_session):
    rows = [{"ticker": "NESN.SW", "drop_pct": -5.2}, {"ticker": "ABBN.SW", "drop_pct": -6.1}]
    scan_persistence.save_rebound_scan(rows, scan_run_at=_dt(2026, 8, 19, 6, 0))

    got_rows, scan_run_at, failed = scan_persistence.get_latest_rebound_scan()
    assert {r["ticker"] for r in got_rows} == {"NESN.SW", "ABBN.SW"}
    assert scan_run_at == _dt(2026, 8, 19, 6, 0)
    assert failed == []


def test_save_rebound_scan_persists_failed_tickers(sqlite_session):
    scan_persistence.save_rebound_scan(
        [{"ticker": "NESN.SW"}], failed_tickers=["ZURN.SW", "UBSG.SW"], scan_run_at=_dt(2026, 8, 19, 6, 0),
    )
    _rows, _scan_run_at, failed = scan_persistence.get_latest_rebound_scan()
    assert failed == ["ZURN.SW", "UBSG.SW"]


def test_save_rebound_scan_replaces_older_rows(sqlite_session):
    scan_persistence.save_rebound_scan([{"ticker": "OLD.SW"}], scan_run_at=_dt(2026, 8, 18, 6, 0))
    scan_persistence.save_rebound_scan([{"ticker": "NEW.SW"}], scan_run_at=_dt(2026, 8, 19, 6, 0))

    rows, scan_run_at, _failed = scan_persistence.get_latest_rebound_scan()
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

    _rows, scan_run_at, _failed = scan_persistence.get_latest_rebound_scan()
    assert before <= scan_run_at.replace(tzinfo=None) <= after


def test_merge_rebound_retry_rows_appends_without_deleting_existing(sqlite_session):
    scan_run_at = _dt(2026, 8, 19, 6, 0)
    scan_persistence.save_rebound_scan(
        [{"ticker": "NESN.SW"}], failed_tickers=["ZURN.SW"], scan_run_at=scan_run_at,
    )

    scan_persistence.merge_rebound_retry_rows(scan_run_at, [{"ticker": "ZURN.SW"}])

    rows, got_scan_run_at, _failed = scan_persistence.get_latest_rebound_scan()
    assert {r["ticker"] for r in rows} == {"NESN.SW", "ZURN.SW"}
    assert got_scan_run_at == scan_run_at


def test_merge_rebound_retry_rows_noop_on_empty_list(sqlite_session):
    scan_run_at = _dt(2026, 8, 19, 6, 0)
    scan_persistence.save_rebound_scan([{"ticker": "NESN.SW"}], scan_run_at=scan_run_at)

    scan_persistence.merge_rebound_retry_rows(scan_run_at, [])

    rows, _scan_run_at, _failed = scan_persistence.get_latest_rebound_scan()
    assert [r["ticker"] for r in rows] == ["NESN.SW"]


def test_merge_rebound_retry_rows_is_idempotent_after_a_crash_before_failed_tickers_update(sqlite_session):
    # Regression: merge_rebound_retry_rows and the subsequent
    # update_rebound_run_failed_tickers call commit in SEPARATE sessions
    # (see scheduler._retry_failed_rebound_tickers) - a crash between the
    # two used to leave a ticker both persisted here AND still listed as
    # failed, so the NEXT retry re-fetched and re-merged the SAME ticker,
    # inserting a second row for it under the same scan_run_at instead of
    # replacing the first.
    scan_run_at = _dt(2026, 8, 19, 6, 0)
    scan_persistence.save_rebound_scan(
        [{"ticker": "NESN.SW"}], failed_tickers=["ZURN.SW"], scan_run_at=scan_run_at,
    )

    # First retry recovers ZURN.SW - simulate the crash by never calling
    # update_rebound_run_failed_tickers afterwards.
    scan_persistence.merge_rebound_retry_rows(scan_run_at, [{"ticker": "ZURN.SW", "drop_pct": -4.0}])

    # Next retry still thinks ZURN.SW is failing (failed_tickers was never
    # updated) and re-merges a fresh row for it.
    scan_persistence.merge_rebound_retry_rows(scan_run_at, [{"ticker": "ZURN.SW", "drop_pct": -4.5}])

    rows, _scan_run_at, _failed = scan_persistence.get_latest_rebound_scan()
    zurn_rows = [r for r in rows if r["ticker"] == "ZURN.SW"]
    assert len(zurn_rows) == 1  # not duplicated
    assert zurn_rows[0]["drop_pct"] == -4.5  # latest merge wins
    assert {r["ticker"] for r in rows} == {"NESN.SW", "ZURN.SW"}


def test_update_rebound_run_failed_tickers_overwrites_the_list(sqlite_session):
    scan_run_at = _dt(2026, 8, 19, 6, 0)
    scan_persistence.save_rebound_scan(
        [{"ticker": "NESN.SW"}], failed_tickers=["ZURN.SW", "UBSG.SW"], scan_run_at=scan_run_at,
    )

    scan_persistence.update_rebound_run_failed_tickers(scan_run_at, ["UBSG.SW"])

    _rows, _scan_run_at, failed = scan_persistence.get_latest_rebound_scan()
    assert failed == ["UBSG.SW"]


# --- volatility indicator ---


def test_get_latest_indicator_scan_returns_empty_for_unscanned_threshold(sqlite_session):
    rows, scan_run_at, failed = scan_persistence.get_latest_indicator_scan(3.0)
    assert rows == []
    assert scan_run_at is None
    assert failed == []


def test_save_and_get_latest_indicator_scan_scoped_per_threshold(sqlite_session):
    scan_persistence.save_indicator_scan(
        [{"ticker": "A.SW"}], threshold_pct=2.0, failed_tickers=["ZURN.SW"], scan_run_at=_dt(2026, 8, 1, 4, 0),
    )
    scan_persistence.save_indicator_scan(
        [{"ticker": "B.SW"}], threshold_pct=5.0, failed_tickers=["ZURN.SW"], scan_run_at=_dt(2026, 8, 1, 4, 0),
    )

    rows_2, run_at_2, failed_2 = scan_persistence.get_latest_indicator_scan(2.0)
    rows_5, run_at_5, failed_5 = scan_persistence.get_latest_indicator_scan(5.0)
    assert [r["ticker"] for r in rows_2] == ["A.SW"]
    assert [r["ticker"] for r in rows_5] == ["B.SW"]
    assert run_at_2 == run_at_5 == _dt(2026, 8, 1, 4, 0)
    # failed_tickers is shared across thresholds for the same run (one
    # discovery pass covers all three) - both reads see the same list,
    # and saving the SAME scan_run_at twice (once per threshold) doesn't
    # duplicate or error on the shared run-meta row.
    assert failed_2 == failed_5 == ["ZURN.SW"]


def test_save_indicator_scan_replaces_older_rows_for_same_threshold_only(sqlite_session):
    scan_persistence.save_indicator_scan([{"ticker": "OLD.SW"}], threshold_pct=2.0, scan_run_at=_dt(2026, 7, 1, 4, 0))
    scan_persistence.save_indicator_scan([{"ticker": "KEEP.SW"}], threshold_pct=5.0, scan_run_at=_dt(2026, 7, 1, 4, 0))
    scan_persistence.save_indicator_scan([{"ticker": "NEW.SW"}], threshold_pct=2.0, scan_run_at=_dt(2026, 8, 1, 4, 0))

    rows_2, _run_at, _failed = scan_persistence.get_latest_indicator_scan(2.0)
    rows_5, _run_at, _failed = scan_persistence.get_latest_indicator_scan(5.0)
    assert [r["ticker"] for r in rows_2] == ["NEW.SW"]
    assert [r["ticker"] for r in rows_5] == ["KEEP.SW"]  # untouched - different threshold


def test_merge_indicator_retry_rows_appends_for_one_threshold(sqlite_session):
    scan_run_at = _dt(2026, 8, 1, 4, 0)
    scan_persistence.save_indicator_scan(
        [{"ticker": "A.SW"}], threshold_pct=2.0, failed_tickers=["ZURN.SW"], scan_run_at=scan_run_at,
    )

    scan_persistence.merge_indicator_retry_rows(scan_run_at, 2.0, [{"ticker": "ZURN.SW"}])

    rows, _run_at, _failed = scan_persistence.get_latest_indicator_scan(2.0)
    assert {r["ticker"] for r in rows} == {"A.SW", "ZURN.SW"}


def test_merge_indicator_retry_rows_noop_on_empty_list(sqlite_session):
    scan_run_at = _dt(2026, 8, 1, 4, 0)
    scan_persistence.save_indicator_scan([{"ticker": "A.SW"}], threshold_pct=2.0, scan_run_at=scan_run_at)

    scan_persistence.merge_indicator_retry_rows(scan_run_at, 2.0, [])

    rows, _run_at, _failed = scan_persistence.get_latest_indicator_scan(2.0)
    assert [r["ticker"] for r in rows] == ["A.SW"]


def test_merge_indicator_retry_rows_is_idempotent_after_a_crash_before_failed_tickers_update(sqlite_session):
    # Same regression as test_merge_rebound_retry_rows_is_idempotent_
    # after_a_crash_before_failed_tickers_update, for the indicator-scan
    # counterpart (merge_indicator_retry_rows / scheduler._retry_failed_
    # indicator_tickers).
    scan_run_at = _dt(2026, 8, 1, 4, 0)
    scan_persistence.save_indicator_scan(
        [{"ticker": "A.SW"}], threshold_pct=2.0, failed_tickers=["ZURN.SW"], scan_run_at=scan_run_at,
    )

    scan_persistence.merge_indicator_retry_rows(scan_run_at, 2.0, [{"ticker": "ZURN.SW", "loss_days": 3}])
    scan_persistence.merge_indicator_retry_rows(scan_run_at, 2.0, [{"ticker": "ZURN.SW", "loss_days": 4}])

    rows, _run_at, _failed = scan_persistence.get_latest_indicator_scan(2.0)
    zurn_rows = [r for r in rows if r["ticker"] == "ZURN.SW"]
    assert len(zurn_rows) == 1  # not duplicated
    assert zurn_rows[0]["loss_days"] == 4  # latest merge wins


def test_update_indicator_run_failed_tickers_overwrites_the_shared_list(sqlite_session):
    scan_run_at = _dt(2026, 8, 1, 4, 0)
    scan_persistence.save_indicator_scan(
        [{"ticker": "A.SW"}], threshold_pct=2.0, failed_tickers=["ZURN.SW", "UBSG.SW"], scan_run_at=scan_run_at,
    )
    scan_persistence.save_indicator_scan(
        [{"ticker": "B.SW"}], threshold_pct=5.0, failed_tickers=["ZURN.SW", "UBSG.SW"], scan_run_at=scan_run_at,
    )

    scan_persistence.update_indicator_run_failed_tickers(scan_run_at, ["UBSG.SW"])

    _rows, _run_at, failed_2 = scan_persistence.get_latest_indicator_scan(2.0)
    _rows, _run_at, failed_5 = scan_persistence.get_latest_indicator_scan(5.0)
    assert failed_2 == failed_5 == ["UBSG.SW"]

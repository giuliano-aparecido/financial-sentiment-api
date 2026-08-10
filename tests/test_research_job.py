import time

import pandas as pd
import pytest

import app.services.research_job as research_job


@pytest.fixture(autouse=True)
def reset_job_state():
    # research_job's job state is module-level (see its own docstring for
    # why - single global slot, not a per-test/per-request object) so
    # tests must reset it before/after each run or they'd see whatever
    # state a previous test left behind.
    research_job._job = {"status": "idle"}
    yield
    research_job._job = {"status": "idle"}


def _patch_scan(monkeypatch, *, crash_rebound_rows=None, today_rows=None, delay=0.0, raises=None):
    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: {"NVDA.SW": {}})
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", lambda candidates: {"NVDA.SW": {}})

    def fake_crash_rebound(domestic):
        if delay:
            time.sleep(delay)
        if raises:
            raise raises
        return pd.DataFrame(crash_rebound_rows or [])

    def fake_today(domestic):
        return pd.DataFrame(today_rows or [])

    monkeypatch.setattr(research_job.swiss_small_cap_crash_rebound, "run_scan", fake_crash_rebound)
    monkeypatch.setattr(research_job.swiss_small_cap_today_screener, "run_scan", fake_today)


def test_get_status_idle_before_any_scan():
    assert research_job.get_status() == {"status": "idle"}


def test_start_scan_returns_running_immediately(monkeypatch):
    _patch_scan(monkeypatch, delay=0.2)
    result = research_job.start_scan()
    assert result["status"] == "running"
    assert "started_at" in result


def test_start_scan_twice_while_running_returns_same_job_not_a_new_one(monkeypatch):
    _patch_scan(monkeypatch, delay=0.3)
    first = research_job.start_scan()
    second = research_job.start_scan()
    assert first == second


def test_scan_completes_and_status_reflects_results(monkeypatch):
    _patch_scan(
        monkeypatch,
        crash_rebound_rows=[{"ticker": "NVDA.SW", "drop_pct": -5.5}],
        today_rows=[{"ticker": "NVDA.SW", "change_pct": -6.0}],
    )
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["status"] == "done"
    assert status["universe_size"] == 1
    assert status["crash_rebound"] == [{"ticker": "NVDA.SW", "drop_pct": -5.5}]
    assert status["today_screener"] == [{"ticker": "NVDA.SW", "change_pct": -6.0}]
    assert "finished_at" in status


def test_scan_failure_sets_error_status(monkeypatch):
    _patch_scan(monkeypatch, raises=RuntimeError("yfinance is down"))
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["status"] == "error"
    assert "yfinance is down" in status["error"]


def test_json_safe_records_converts_nan_to_none():
    df = pd.DataFrame([{"a": 1.0, "b": "x"}, {"a": float("nan"), "b": "y"}])
    records = research_job._json_safe_records(df)
    assert records[0] == {"a": 1.0, "b": "x"}
    assert records[1]["a"] is None
    assert records[1]["b"] == "y"


def test_json_safe_records_empty_dataframe_returns_empty_list():
    assert research_job._json_safe_records(pd.DataFrame()) == []


def test_scan_with_missing_pe_and_volume_ratio_serializes_as_json_null(monkeypatch):
    # Confirmed live: df.to_dict(orient="records") alone leaves NaN as the
    # Python float nan, which json.dumps renders as the invalid bare
    # token NaN - this test guards the fix (_json_safe_records) rather
    # than re-testing pandas' own NaN behavior.
    _patch_scan(
        monkeypatch,
        crash_rebound_rows=[{"ticker": "NVDA.SW", "loss_pe_approx": float("nan")}],
        today_rows=[],
    )
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["crash_rebound"][0]["loss_pe_approx"] is None
    import json
    json.dumps(status)  # must not raise / must not embed a bare NaN token


def test_empty_results_return_empty_lists_not_missing_keys(monkeypatch):
    _patch_scan(monkeypatch, crash_rebound_rows=[], today_rows=[])
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["crash_rebound"] == []
    assert status["today_screener"] == []

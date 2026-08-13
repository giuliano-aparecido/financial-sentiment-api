import datetime
import time

import pandas as pd
import pytest

import app.services.research_job as research_job


@pytest.fixture(autouse=True)
def reset_job_state():
    # research_job's job state and crash-rebound cache are both
    # module-level (see the module's own docstring for why - single
    # global slot, not a per-test/per-request object) so tests must reset
    # them before/after each run or they'd see whatever state a previous
    # test left behind.
    research_job._job = {"status": "idle"}
    research_job._crash_rebound_cache = {"date": None, "all_caps": None, "result": None}
    yield
    research_job._job = {"status": "idle"}
    research_job._crash_rebound_cache = {"date": None, "all_caps": None, "result": None}


class _CallList(list):
    """Plain list subclass so a second call log (discover_calls) can be
    attached as an attribute without changing _patch_scan's return type
    for the many existing tests that do `calls = _patch_scan(...)`."""


def _patch_scan(monkeypatch, *, crash_rebound_rows=None, today_rows=None, delay=0.0, raises=None):
    discover_calls = []

    def fake_discover(**kwargs):
        discover_calls.append(kwargs)
        return {"NVDA.SW": {}}

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", fake_discover)
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", lambda candidates: {"NVDA.SW": {}})

    crash_rebound_calls = _CallList()

    def fake_crash_rebound(domestic):
        crash_rebound_calls.append(domestic)
        if delay:
            time.sleep(delay)
        if raises:
            raise raises
        return pd.DataFrame(crash_rebound_rows or [])

    def fake_today(domestic):
        return pd.DataFrame(today_rows or [])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_crash_rebound)
    monkeypatch.setattr(research_job.swiss_today_screener, "run_scan", fake_today)
    crash_rebound_calls.discover_calls = discover_calls  # exposed for all_caps-threading tests
    return crash_rebound_calls


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


# --- crash-rebound daily cache ---
# Its 3-month lookback barely changes day to day (a closed trading day's
# OHLCV doesn't change), so it's recomputed at most once per calendar day
# (see research_job.py's module docstring) - today_screener is NEVER
# cached, it always re-runs for live intraday quotes.


def test_crash_rebound_result_starts_uncached():
    assert research_job._crash_rebound_cache == {"date": None, "all_caps": None, "result": None}


def test_crash_rebound_result_reuses_cache_within_same_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._crash_rebound_result({"A.SW": {}}, all_caps=False)
    second = research_job._crash_rebound_result({"A.SW": {}}, all_caps=False)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["call_number"] == 1


def test_crash_rebound_result_recomputes_on_a_new_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))
    first = research_job._crash_rebound_result({"A.SW": {}}, all_caps=False)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    second = research_job._crash_rebound_result({"A.SW": {}}, all_caps=False)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


def test_crash_rebound_result_recomputes_when_all_caps_mode_changes_same_day(monkeypatch):
    # Regression guard: a small-cap scan and an all-caps scan cover
    # different universes, so switching modes within the same UTC day
    # must NOT reuse the other mode's cached result.
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    small_cap_result = research_job._crash_rebound_result({"A.SW": {}}, all_caps=False)
    all_caps_result = research_job._crash_rebound_result({"A.SW": {}}, all_caps=True)

    assert len(calls) == 2
    assert small_cap_result.iloc[0]["call_number"] == 1
    assert all_caps_result.iloc[0]["call_number"] == 2


# --- all_caps parameter threading ---


def test_start_scan_defaults_to_small_cap_band(monkeypatch):
    calls = _patch_scan(monkeypatch)
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["all_caps"] is False
    # No kwargs -> same call shape discover_candidates() had before
    # all_caps existed, so its own MIN/MAX_MARKET_CAP_CHF defaults apply.
    assert calls.discover_calls == [{}]


def test_start_scan_all_caps_true_passes_wide_market_cap_band(monkeypatch):
    calls = _patch_scan(monkeypatch)
    research_job.start_scan(all_caps=True)

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["all_caps"] is True
    assert calls.discover_calls == [{
        "min_market_cap": research_job.swiss_universe.ALL_CAPS_MIN_MARKET_CAP_CHF,
        "max_market_cap": research_job.swiss_universe.ALL_CAPS_MAX_MARKET_CAP_CHF,
    }]


def test_start_scan_running_job_reports_its_own_all_caps_value(monkeypatch):
    _patch_scan(monkeypatch, delay=0.3)
    result = research_job.start_scan(all_caps=True)
    assert result["all_caps"] is True


def test_scan_failure_status_includes_all_caps(monkeypatch):
    _patch_scan(monkeypatch, raises=RuntimeError("yfinance is down"))
    research_job.start_scan(all_caps=True)

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["status"] == "error"
    assert status["all_caps"] is True


def test_full_scan_via_start_scan_uses_cache_on_second_same_day_run(monkeypatch):
    calls = _patch_scan(monkeypatch, crash_rebound_rows=[{"ticker": "NVDA.SW"}], today_rows=[])
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    def run_and_wait():
        research_job.start_scan()
        deadline = time.time() + 2
        status = research_job.get_status()
        while status["status"] == "running" and time.time() < deadline:
            time.sleep(0.02)
            status = research_job.get_status()
        return status

    first_status = run_and_wait()
    # Reset job status back to idle so the second call doesn't just return
    # the (still "done") in-flight job unchanged - start_scan only skips
    # starting a NEW scan while one is "running" (see its own docstring).
    research_job._job = {"status": "idle"}
    second_status = run_and_wait()

    assert len(calls) == 1  # crash_rebound.run_scan only called once
    assert first_status["crash_rebound"] == second_status["crash_rebound"] == [{"ticker": "NVDA.SW"}]

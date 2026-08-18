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
    research_job._crash_rebound_cache = {"date": None, "result": None}
    research_job._indicator_job = {"status": "idle"}
    research_job._indicator_cache = {"date": None, "threshold_pct": None, "result": None}
    yield
    research_job._job = {"status": "idle"}
    research_job._crash_rebound_cache = {"date": None, "result": None}
    research_job._indicator_job = {"status": "idle"}
    research_job._indicator_cache = {"date": None, "threshold_pct": None, "result": None}


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
    crash_rebound_calls.discover_calls = discover_calls  # exposed for discover_candidates()-call-shape tests
    return crash_rebound_calls


def test_get_status_idle_before_any_scan():
    assert research_job.get_status() == {"status": "idle"}


def _wait_until_not_running(deadline_seconds=2):
    # Drains the background thread started by start_scan() before the
    # calling test returns - without this, a test that only checks the
    # immediate "running" state leaves that thread executing past the end
    # of the test function, where it keeps referencing whatever this
    # test's own `monkeypatch` fixture patched (research_job.
    # swiss_today_screener.run_scan, etc.) - patches pytest reverts the
    # MOMENT the test function returns. A slow-enough background thread
    # can end up calling the REAL (un-mocked) function on fake test data
    # after that revert, which is a real, confirmed-live failure mode
    # (KeyError on a fake domestic dict missing real yfinance-shaped
    # fields), not a hypothetical one - draining it here, still under the
    # same test's monkeypatch, avoids that race entirely.
    deadline = time.time() + deadline_seconds
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()
    return status


def test_start_scan_returns_running_immediately(monkeypatch):
    _patch_scan(monkeypatch, delay=0.2)
    result = research_job.start_scan()
    assert result["status"] == "running"
    assert "started_at" in result
    _wait_until_not_running()


def test_start_scan_twice_while_running_returns_same_job_not_a_new_one(monkeypatch):
    _patch_scan(monkeypatch, delay=0.3)
    first = research_job.start_scan()
    second = research_job.start_scan()
    assert first == second
    _wait_until_not_running()


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
# Its 12-month lookback barely changes day to day (a closed trading day's
# OHLCV doesn't change), so it's recomputed at most once per calendar day
# (see research_job.py's module docstring) - today_screener is NEVER
# cached, it always re-runs for live intraday quotes.


def test_crash_rebound_result_starts_uncached():
    assert research_job._crash_rebound_cache == {"date": None, "result": None}


def test_crash_rebound_result_reuses_cache_within_same_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._crash_rebound_result({"A.SW": {}})
    second = research_job._crash_rebound_result({"A.SW": {}})

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
    first = research_job._crash_rebound_result({"A.SW": {}})

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    second = research_job._crash_rebound_result({"A.SW": {}})

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


# --- discover_candidates() call shape ---


def test_start_scan_uses_no_kwargs_discovery_call(monkeypatch):
    calls = _patch_scan(monkeypatch)
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["status"] == "done"
    # No kwargs -> discover_candidates()'s own MIN/MAX_MARKET_CAP_CHF
    # defaults apply (single universe, see swiss_universe.py - no more
    # all_caps mode to pass through).
    assert calls.discover_calls == [{}]


def test_scan_failure_status_has_no_all_caps_key(monkeypatch):
    _patch_scan(monkeypatch, raises=RuntimeError("yfinance is down"))
    research_job.start_scan()

    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()

    assert status["status"] == "error"
    assert "all_caps" not in status


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


# --- volatility-indicator scan (entirely separate job/cache from _job above) ---


def _patch_indicator_scan(monkeypatch, *, rows=None, delay=0.0, raises=None):
    discover_calls = []

    def fake_discover(**kwargs):
        discover_calls.append(kwargs)
        return {"NVDA.SW": {}}

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", fake_discover)
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", lambda candidates: {"NVDA.SW": {}})

    indicator_calls = _CallList()

    def fake_indicator_scan(domestic, threshold_pct):
        indicator_calls.append((domestic, threshold_pct))
        if delay:
            time.sleep(delay)
        if raises:
            raise raises
        return pd.DataFrame(rows or [])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_indicator_scan)
    indicator_calls.discover_calls = discover_calls
    return indicator_calls


def _wait_until_indicator_not_running(deadline_seconds=2):
    deadline = time.time() + deadline_seconds
    status = research_job.get_indicator_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_indicator_status()
    return status


def test_get_indicator_status_idle_before_any_scan():
    assert research_job.get_indicator_status() == {"status": "idle"}


def test_start_indicator_scan_returns_running_immediately(monkeypatch):
    _patch_indicator_scan(monkeypatch, delay=0.2)
    result = research_job.start_indicator_scan(2.0)
    assert result["status"] == "running"
    assert result["threshold_pct"] == 2.0
    _wait_until_indicator_not_running()


def test_start_indicator_scan_twice_while_running_returns_same_job(monkeypatch):
    _patch_indicator_scan(monkeypatch, delay=0.3)
    first = research_job.start_indicator_scan(2.0)
    second = research_job.start_indicator_scan(5.0)  # ignored while running
    assert first == second
    _wait_until_indicator_not_running()


def test_indicator_scan_completes_and_status_reflects_results(monkeypatch):
    _patch_indicator_scan(monkeypatch, rows=[{"ticker": "NVDA.SW", "total_days": 4}])
    research_job.start_indicator_scan(2.0)
    status = _wait_until_indicator_not_running()

    assert status["status"] == "done"
    assert status["threshold_pct"] == 2.0
    assert status["universe_size"] == 1
    assert status["volatility_indicator"] == [{"ticker": "NVDA.SW", "total_days": 4}]


def test_indicator_scan_failure_sets_error_status(monkeypatch):
    _patch_indicator_scan(monkeypatch, raises=RuntimeError("yfinance is down"))
    research_job.start_indicator_scan(3.0)
    status = _wait_until_indicator_not_running()

    assert status["status"] == "error"
    assert status["threshold_pct"] == 3.0
    assert "yfinance is down" in status["error"]


def test_indicator_scan_uses_no_kwargs_discovery_call(monkeypatch):
    calls = _patch_indicator_scan(monkeypatch)
    research_job.start_indicator_scan(2.0)
    _wait_until_indicator_not_running()
    # Independent discovery (see swiss_volatility_indicator.py's own
    # docstring for why it doesn't reuse the main scan's `domestic`).
    assert calls.discover_calls == [{}]


def test_indicator_result_reuses_cache_within_same_day_and_threshold(monkeypatch):
    calls = []

    def fake_run_scan(domestic, threshold_pct):
        calls.append((domestic, threshold_pct))
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._indicator_result({"A.SW": {}}, 2.0)
    second = research_job._indicator_result({"A.SW": {}}, 2.0)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_indicator_result_recomputes_when_threshold_changes_same_day(monkeypatch):
    # Regression guard: 2%/3%/5% are genuinely different scans over the
    # same universe, not one scan with a display-only filter - switching
    # thresholds within the same UTC day must NOT reuse the other
    # threshold's cached result.
    calls = []

    def fake_run_scan(domestic, threshold_pct):
        calls.append((domestic, threshold_pct))
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    two_pct_result = research_job._indicator_result({"A.SW": {}}, 2.0)
    five_pct_result = research_job._indicator_result({"A.SW": {}}, 5.0)

    assert len(calls) == 2
    assert two_pct_result.iloc[0]["call_number"] == 1
    assert five_pct_result.iloc[0]["call_number"] == 2


def test_indicator_result_recomputes_on_a_new_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic, threshold_pct):
        calls.append((domestic, threshold_pct))
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_run_scan)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))
    first = research_job._indicator_result({"A.SW": {}}, 2.0)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    second = research_job._indicator_result({"A.SW": {}}, 2.0)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


def test_indicator_scan_is_independent_of_main_scan_state(monkeypatch):
    # The two job slots must not interfere with each other - starting one
    # while the other is running/done shouldn't affect its status.
    _patch_scan(monkeypatch, crash_rebound_rows=[{"ticker": "NVDA.SW"}], today_rows=[])
    research_job.start_scan()
    deadline = time.time() + 2
    status = research_job.get_status()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = research_job.get_status()
    assert status["status"] == "done"

    assert research_job.get_indicator_status() == {"status": "idle"}

    _patch_indicator_scan(monkeypatch, rows=[{"ticker": "NVDA.SW", "total_days": 2}])
    research_job.start_indicator_scan(2.0)
    indicator_status = _wait_until_indicator_not_running()

    assert indicator_status["status"] == "done"
    # Main scan's own status untouched by the indicator scan running.
    assert research_job.get_status()["status"] == "done"

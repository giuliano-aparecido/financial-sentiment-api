import datetime
import time

import pandas as pd
import pytest

import app.services.research_job as research_job


@pytest.fixture(autouse=True)
def reset_job_state():
    # Each scan type's job/cache state is module-level (see the module's
    # own docstring for why - single global slot per scan TYPE, not a
    # per-test/per-request object) so tests must reset them before/after
    # each run or they'd see whatever state a previous test left behind.
    research_job._rebound_slot = research_job._JobSlot()
    research_job._rebound_cache = {"date": None, "result": None}
    research_job._rebound_discovery_cache = research_job._new_discovery_cache()
    research_job._today_slot = research_job._JobSlot()
    research_job._today_discovery_cache = research_job._new_discovery_cache()
    research_job._indicator_slot = research_job._JobSlot()
    research_job._indicator_cache = {"date": None, "threshold_pct": None, "result": None}
    research_job._indicator_discovery_cache = research_job._new_discovery_cache()
    yield
    research_job._rebound_slot = research_job._JobSlot()
    research_job._rebound_cache = {"date": None, "result": None}
    research_job._rebound_discovery_cache = research_job._new_discovery_cache()
    research_job._today_slot = research_job._JobSlot()
    research_job._today_discovery_cache = research_job._new_discovery_cache()
    research_job._indicator_slot = research_job._JobSlot()
    research_job._indicator_cache = {"date": None, "threshold_pct": None, "result": None}
    research_job._indicator_discovery_cache = research_job._new_discovery_cache()


def _wait_until(status_fn, deadline_seconds=2):
    deadline = time.time() + deadline_seconds
    status = status_fn()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = status_fn()
    return status


def _patch_discovery(monkeypatch, domestic=None, failed_symbols=None):
    discover_calls = []

    def fake_discover(**kwargs):
        discover_calls.append(kwargs)
        return {"NVDA.SW": {}}

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", fake_discover)
    monkeypatch.setattr(
        research_job.swiss_universe, "filter_domestic",
        lambda candidates: (domestic or {"NVDA.SW": {}}, failed_symbols or []),
    )
    return discover_calls


# --- _JobSlot (shared single-flight helper) ---


def test_job_slot_starts_idle():
    slot = research_job._JobSlot()
    assert slot.status() == {"status": "idle"}


def test_job_slot_start_returns_running_immediately():
    slot = research_job._JobSlot()

    def slow_run(s, started_at):
        time.sleep(0.2)
        s.set_done(started_at, result="ok")

    result = slot.start(slow_run)
    assert result["status"] == "running"
    assert "started_at" in result
    _wait_until(slot.status)


def test_job_slot_start_twice_while_running_returns_same_job():
    slot = research_job._JobSlot()

    def slow_run(s, started_at):
        time.sleep(0.3)
        s.set_done(started_at, result="ok")

    first = slot.start(slow_run)
    second = slot.start(slow_run)
    assert first == second
    _wait_until(slot.status)


def test_job_slot_start_accepts_running_fields():
    slot = research_job._JobSlot()

    def slow_run(s, started_at, x):
        time.sleep(0.1)
        s.set_done(started_at, x=x)

    result = slot.start(slow_run, 7, x=7)
    assert result["x"] == 7
    _wait_until(slot.status)


def test_job_slot_set_done_and_set_error():
    slot = research_job._JobSlot()
    slot.set_done("t0", foo="bar")
    assert slot.status() == {"status": "done", "started_at": "t0", "finished_at": slot.job["finished_at"], "foo": "bar"}

    slot.set_error("t0", "boom", foo="bar")
    assert slot.status()["status"] == "error"
    assert slot.status()["error"] == "boom"
    assert slot.status()["foo"] == "bar"


# --- Discovery caching + partial-failure retry ---


def test_discover_and_filter_fresh_day_runs_full_discovery(monkeypatch):
    filter_calls = []

    def fake_filter(candidates):
        filter_calls.append(candidates)
        return {"A.SW": {}, "B.SW": {}}, []

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: {"A.SW": "qa", "B.SW": "qb"})
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", fake_filter)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    cache = research_job._new_discovery_cache()
    domestic, changed = research_job._discover_and_filter_with_retry(cache)

    assert changed is True
    assert domestic == {"A.SW": {}, "B.SW": {}}
    assert len(filter_calls) == 1
    assert cache["date"] == datetime.date(2026, 8, 10)
    assert cache["failed_symbols"] == []


def test_discover_and_filter_same_day_no_failures_is_pure_cache_hit(monkeypatch):
    discover_calls = []
    filter_calls = []

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: (discover_calls.append(1), {"A.SW": "qa"})[1])
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", lambda candidates: (filter_calls.append(candidates), ({"A.SW": {}}, []))[1])
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    cache = research_job._new_discovery_cache()
    research_job._discover_and_filter_with_retry(cache)
    domestic, changed = research_job._discover_and_filter_with_retry(cache)

    assert changed is False
    assert domestic == {"A.SW": {}}
    assert len(discover_calls) == 1
    assert len(filter_calls) == 1  # only the first (fresh-day) call, no re-filter on the cache-hit call


def test_discover_and_filter_same_day_retries_only_previously_failed_symbols(monkeypatch):
    filter_calls = []

    def fake_filter(candidates):
        filter_calls.append(dict(candidates))
        if len(filter_calls) == 1:
            return {"A.SW": {}}, ["B.SW"]  # B.SW fails on the initial discovery pass
        return {"B.SW": {}}, []  # B.SW succeeds on retry

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: {"A.SW": "qa", "B.SW": "qb"})
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", fake_filter)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    cache = research_job._new_discovery_cache()
    first_domestic, first_changed = research_job._discover_and_filter_with_retry(cache)
    # cache["domestic"] is mutated in place and returned by reference, not
    # copied - snapshot it now, before the retry call below updates it, or
    # this assertion would observe the LATER state through the same dict.
    first_domestic = dict(first_domestic)
    second_domestic, second_changed = research_job._discover_and_filter_with_retry(cache)

    assert first_domestic == {"A.SW": {}}
    assert first_changed is True
    assert len(filter_calls) == 2
    assert filter_calls[1] == {"B.SW": "qb"}  # retry call only re-fetches the one failed symbol
    assert second_domestic == {"A.SW": {}, "B.SW": {}}  # newly-recovered ticker merged in permanently
    assert second_changed is True
    assert cache["failed_symbols"] == []


def test_discover_and_filter_still_failing_symbols_stay_for_next_retry(monkeypatch):
    filter_calls = []

    def fake_filter(candidates):
        filter_calls.append(dict(candidates))
        if len(filter_calls) == 1:
            return {"A.SW": {}}, ["B.SW"]
        return {}, ["B.SW"]  # still failing on retry

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: {"A.SW": "qa", "B.SW": "qb"})
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", fake_filter)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    cache = research_job._new_discovery_cache()
    research_job._discover_and_filter_with_retry(cache)
    second_domestic, second_changed = research_job._discover_and_filter_with_retry(cache)

    assert second_domestic == {"A.SW": {}}  # no new recovery - domestic unchanged
    assert second_changed is False
    assert cache["failed_symbols"] == ["B.SW"]


def test_discover_and_filter_new_day_rediscovers_full_universe_even_with_leftover_failures(monkeypatch):
    filter_calls = []

    def fake_filter(candidates):
        filter_calls.append(dict(candidates))
        return {"A.SW": {}}, ["B.SW"]

    monkeypatch.setattr(research_job.swiss_universe, "discover_candidates", lambda: {"A.SW": "qa", "B.SW": "qb"})
    monkeypatch.setattr(research_job.swiss_universe, "filter_domestic", fake_filter)

    cache = research_job._new_discovery_cache()
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))
    research_job._discover_and_filter_with_retry(cache)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    domestic, changed = research_job._discover_and_filter_with_retry(cache)

    assert changed is True
    assert len(filter_calls) == 2
    assert filter_calls[1] == {"A.SW": "qa", "B.SW": "qb"}  # full re-discovery, not just the failed symbol
    assert cache["date"] == datetime.date(2026, 8, 11)


def test_rebound_scan_surfaces_failed_ticker_count_from_discovery(monkeypatch):
    _patch_discovery(monkeypatch, failed_symbols=["ZUGN.SW"])
    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", lambda domestic: pd.DataFrame())
    research_job.start_rebound_scan()
    status = _wait_until(research_job.get_rebound_status)
    assert status["failed_ticker_count"] == 1


# --- Rebound scan ---


def test_get_rebound_status_idle_before_any_scan():
    assert research_job.get_rebound_status() == {"status": "idle"}


def test_rebound_scan_completes_and_status_reflects_results(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NVDA.SW", "drop_pct": -5.5}]),
    )
    research_job.start_rebound_scan()
    status = _wait_until(research_job.get_rebound_status)

    assert status["status"] == "done"
    assert status["universe_size"] == 1
    assert status["failed_ticker_count"] == 0
    assert status["crash_rebound"] == [{"ticker": "NVDA.SW", "drop_pct": -5.5}]
    assert "finished_at" in status


def test_rebound_scan_failure_sets_error_status(monkeypatch):
    _patch_discovery(monkeypatch)

    def raises(domestic):
        raise RuntimeError("yfinance is down")

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", raises)
    research_job.start_rebound_scan()
    status = _wait_until(research_job.get_rebound_status)

    assert status["status"] == "error"
    assert "yfinance is down" in status["error"]


def test_rebound_scan_uses_no_kwargs_discovery_call(monkeypatch):
    calls = _patch_discovery(monkeypatch)
    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", lambda domestic: pd.DataFrame())
    research_job.start_rebound_scan()
    _wait_until(research_job.get_rebound_status)
    assert calls == [{}]


def test_rebound_result_reuses_cache_within_same_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._rebound_result({"A.SW": {}}, force=False)
    second = research_job._rebound_result({"A.SW": {}}, force=False)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_rebound_result_force_bypasses_cache_within_same_day(monkeypatch):
    # Regression guard for the discovery-cache-with-retry fix: a same-day
    # retry that recovers previously-failed tickers must NOT keep serving
    # the stale, smaller-universe cached result (see
    # _discover_and_filter_with_retry's own docstring for the full "why").
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._rebound_result({"A.SW": {}}, force=False)
    second = research_job._rebound_result({"A.SW": {}, "B.SW": {}}, force=True)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


def test_rebound_result_recomputes_on_a_new_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))
    first = research_job._rebound_result({"A.SW": {}}, force=False)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    second = research_job._rebound_result({"A.SW": {}}, force=False)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


def test_rebound_scan_uses_cache_on_second_same_day_run(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "NVDA.SW"}])

    _patch_discovery(monkeypatch)
    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    research_job.start_rebound_scan()
    first_status = _wait_until(research_job.get_rebound_status)
    research_job._rebound_slot = research_job._JobSlot()  # simulate a fresh page load's own status check
    research_job.start_rebound_scan()
    second_status = _wait_until(research_job.get_rebound_status)

    assert len(calls) == 1
    assert first_status["crash_rebound"] == second_status["crash_rebound"] == [{"ticker": "NVDA.SW"}]


def test_rebound_scan_independent_of_today_and_indicator_state(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NVDA.SW"}]),
    )
    research_job.start_rebound_scan()
    _wait_until(research_job.get_rebound_status)

    assert research_job.get_today_status() == {"status": "idle"}
    assert research_job.get_indicator_status() == {"status": "idle"}


# --- Today (big-loss) scan ---


def test_get_today_status_idle_before_any_scan():
    assert research_job.get_today_status() == {"status": "idle"}


def test_today_scan_completes_and_status_reflects_results(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_today_screener, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NVDA.SW", "change_pct": -6.0}]),
    )
    research_job.start_today_scan()
    status = _wait_until(research_job.get_today_status)

    assert status["status"] == "done"
    assert status["universe_size"] == 1
    assert status["failed_ticker_count"] == 0
    assert status["today_screener"] == [{"ticker": "NVDA.SW", "change_pct": -6.0}]


def test_today_scan_failure_sets_error_status(monkeypatch):
    _patch_discovery(monkeypatch)

    def raises(domestic):
        raise RuntimeError("yfinance is down")

    monkeypatch.setattr(research_job.swiss_today_screener, "run_scan", raises)
    research_job.start_today_scan()
    status = _wait_until(research_job.get_today_status)

    assert status["status"] == "error"
    assert "yfinance is down" in status["error"]


def test_today_scan_never_cached_recomputes_every_call(monkeypatch):
    calls = []

    def fake_run_scan(domestic):
        calls.append(domestic)
        return pd.DataFrame([{"ticker": "NVDA.SW", "call_number": len(calls)}])

    _patch_discovery(monkeypatch)
    monkeypatch.setattr(research_job.swiss_today_screener, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    research_job.start_today_scan()
    first_status = _wait_until(research_job.get_today_status)
    research_job._today_slot = research_job._JobSlot()
    research_job.start_today_scan()
    second_status = _wait_until(research_job.get_today_status)

    assert len(calls) == 2  # no caching - always re-runs
    assert first_status["today_screener"][0]["call_number"] == 1
    assert second_status["today_screener"][0]["call_number"] == 2


def test_json_safe_records_converts_nan_to_none():
    df = pd.DataFrame([{"a": 1.0, "b": "x"}, {"a": float("nan"), "b": "y"}])
    records = research_job._json_safe_records(df)
    assert records[0] == {"a": 1.0, "b": "x"}
    assert records[1]["a"] is None
    assert records[1]["b"] == "y"


def test_json_safe_records_empty_dataframe_returns_empty_list():
    assert research_job._json_safe_records(pd.DataFrame()) == []


def test_rebound_scan_with_missing_pe_serializes_as_json_null(monkeypatch):
    # Confirmed live: df.to_dict(orient="records") alone leaves NaN as the
    # Python float nan, which json.dumps renders as the invalid bare
    # token NaN - this test guards the fix (_json_safe_records) rather
    # than re-testing pandas' own NaN behavior.
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NVDA.SW", "loss_pe_approx": float("nan")}]),
    )
    research_job.start_rebound_scan()
    status = _wait_until(research_job.get_rebound_status)

    assert status["crash_rebound"][0]["loss_pe_approx"] is None
    import json
    json.dumps(status)  # must not raise / must not embed a bare NaN token


def test_rebound_empty_results_return_empty_list_not_missing_key(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(research_job.swiss_crash_rebound, "run_scan", lambda domestic: pd.DataFrame())
    research_job.start_rebound_scan()
    status = _wait_until(research_job.get_rebound_status)
    assert status["crash_rebound"] == []


# --- Volatility-indicator scan (entirely separate job/cache) ---


def test_get_indicator_status_idle_before_any_scan():
    assert research_job.get_indicator_status() == {"status": "idle"}


def test_start_indicator_scan_returns_running_immediately(monkeypatch):
    _patch_discovery(monkeypatch)

    def slow_run(domestic, threshold_pct):
        time.sleep(0.2)
        return pd.DataFrame()

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", slow_run)
    result = research_job.start_indicator_scan(2.0)
    assert result["status"] == "running"
    assert result["threshold_pct"] == 2.0
    _wait_until(research_job.get_indicator_status)


def test_start_indicator_scan_twice_while_running_returns_same_job(monkeypatch):
    _patch_discovery(monkeypatch)

    def slow_run(domestic, threshold_pct):
        time.sleep(0.3)
        return pd.DataFrame()

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", slow_run)
    first = research_job.start_indicator_scan(2.0)
    second = research_job.start_indicator_scan(5.0)  # ignored while running
    assert first == second
    _wait_until(research_job.get_indicator_status)


def test_indicator_scan_completes_and_status_reflects_results(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_volatility_indicator, "run_scan",
        lambda domestic, threshold_pct: pd.DataFrame([{"ticker": "NVDA.SW", "total_days": 4}]),
    )
    research_job.start_indicator_scan(2.0)
    status = _wait_until(research_job.get_indicator_status)

    assert status["status"] == "done"
    assert status["threshold_pct"] == 2.0
    assert status["universe_size"] == 1
    assert status["failed_ticker_count"] == 0
    assert status["volatility_indicator"] == [{"ticker": "NVDA.SW", "total_days": 4}]


def test_indicator_scan_failure_sets_error_status(monkeypatch):
    _patch_discovery(monkeypatch)

    def raises(domestic, threshold_pct):
        raise RuntimeError("yfinance is down")

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", raises)
    research_job.start_indicator_scan(3.0)
    status = _wait_until(research_job.get_indicator_status)

    assert status["status"] == "error"
    assert status["threshold_pct"] == 3.0
    assert "yfinance is down" in status["error"]


def test_indicator_scan_uses_no_kwargs_discovery_call(monkeypatch):
    calls = _patch_discovery(monkeypatch)
    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", lambda domestic, threshold_pct: pd.DataFrame())
    research_job.start_indicator_scan(2.0)
    _wait_until(research_job.get_indicator_status)
    assert calls == [{}]


def test_indicator_result_reuses_cache_within_same_day_and_threshold(monkeypatch):
    calls = []

    def fake_run_scan(domestic, threshold_pct):
        calls.append((domestic, threshold_pct))
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)
    second = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_indicator_result_force_bypasses_cache_within_same_day(monkeypatch):
    calls = []

    def fake_run_scan(domestic, threshold_pct):
        calls.append((domestic, threshold_pct))
        return pd.DataFrame([{"ticker": "A.SW", "call_number": len(calls)}])

    monkeypatch.setattr(research_job.swiss_volatility_indicator, "run_scan", fake_run_scan)
    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 10))

    first = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)
    second = research_job._indicator_result({"A.SW": {}, "B.SW": {}}, 2.0, force=True)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


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

    two_pct_result = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)
    five_pct_result = research_job._indicator_result({"A.SW": {}}, 5.0, force=False)

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
    first = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)

    monkeypatch.setattr(research_job, "_today", lambda: datetime.date(2026, 8, 11))
    second = research_job._indicator_result({"A.SW": {}}, 2.0, force=False)

    assert len(calls) == 2
    assert first.iloc[0]["call_number"] == 1
    assert second.iloc[0]["call_number"] == 2


def test_indicator_scan_is_independent_of_rebound_and_today_state(monkeypatch):
    _patch_discovery(monkeypatch)
    monkeypatch.setattr(
        research_job.swiss_volatility_indicator, "run_scan",
        lambda domestic, threshold_pct: pd.DataFrame([{"ticker": "NVDA.SW", "total_days": 2}]),
    )
    research_job.start_indicator_scan(2.0)
    status = _wait_until(research_job.get_indicator_status)

    assert status["status"] == "done"
    assert research_job.get_rebound_status() == {"status": "idle"}
    assert research_job.get_today_status() == {"status": "idle"}

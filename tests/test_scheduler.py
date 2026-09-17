import datetime
import threading
import time

import pandas as pd

import app.services.scheduler as scheduler_module
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS


def teardown_function(_fn):
    # trigger_*/{_run_*_guarded} mutate _rebound_running/_indicator_running -
    # reset after every test in this file regardless of pass/fail, so one
    # test's state can't leak into the next.
    scheduler_module._rebound_running = False
    scheduler_module._indicator_running = False


# --- _run_rebound_scan / _run_indicator_scans (the actual pipeline) ---


def _fake_domestic():
    return {"NESN.SW": {"name": "Nestle"}, "ABBN.SW": {"name": "ABB"}}


def test_run_rebound_scan_discovers_scans_and_persists(monkeypatch):
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"NESN.SW": {}, "ABBN.SW": {}})
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic_batched",
        lambda candidates, num_batches, batch_delay_seconds: (_fake_domestic(), ["ZURN.SW"], False),
    )
    monkeypatch.setattr(
        scheduler_module.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NESN.SW", "drop_pct": -5.2}]),
    )
    saved = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "save_rebound_scan",
        lambda rows, failed_tickers=None, scan_run_at=None: saved.update(
            rows=rows, failed_tickers=failed_tickers, scan_run_at=scan_run_at,
        ),
    )

    scheduler_module._run_rebound_scan()

    assert saved["rows"] == [{"ticker": "NESN.SW", "drop_pct": -5.2}]
    assert saved["failed_tickers"] == ["ZURN.SW"]


def test_run_indicator_scans_scans_every_allowed_threshold_off_one_discovery(monkeypatch):
    discover_calls = []
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"NESN.SW": {}})
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic_batched",
        lambda candidates, num_batches, batch_delay_seconds: discover_calls.append(1) or (_fake_domestic(), ["ZURN.SW"], False),
    )
    scanned_thresholds = []
    monkeypatch.setattr(
        scheduler_module.swiss_volatility_indicator, "run_scan",
        lambda domestic, threshold_pct: scanned_thresholds.append(threshold_pct) or pd.DataFrame(
            [{"ticker": "NESN.SW", "loss_days": 2, "gain_days": 1}]
        ),
    )
    saved = []
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "save_indicator_scan",
        lambda rows, threshold_pct, failed_tickers=None, scan_run_at=None: saved.append(
            (threshold_pct, rows, failed_tickers, scan_run_at),
        ),
    )

    scheduler_module._run_indicator_scans()

    # One discovery pass shared across all thresholds, not one per threshold.
    assert len(discover_calls) == 1
    assert scanned_thresholds == list(ALLOWED_THRESHOLD_PCTS)
    assert [s[0] for s in saved] == list(ALLOWED_THRESHOLD_PCTS)
    assert all(s[2] == ["ZURN.SW"] for s in saved)
    # All three saved under the SAME scan_run_at - one shared run, not
    # three independently-timestamped ones.
    assert len({s[3] for s in saved}) == 1


# --- Failed-ticker retry (manual trigger only - see module docstring) ---


def test_retry_failed_rebound_tickers_recovers_and_merges(monkeypatch):
    scan_run_at = scheduler_module._now()
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "discover_candidates",
        lambda: {"ZURN.SW": {"q": 1}, "UBSG.SW": {"q": 2}, "OTHER.SW": {"q": 3}},
    )
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic",
        lambda candidates: ({"ZURN.SW": {"name": "Zurich"}}, ["UBSG.SW"], False),
    )
    monkeypatch.setattr(
        scheduler_module.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "ZURN.SW", "drop_pct": -5.0}]),
    )
    merged = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "merge_rebound_retry_rows",
        lambda run_at, rows: merged.update(run_at=run_at, rows=rows),
    )
    updated = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "update_rebound_run_failed_tickers",
        lambda run_at, failed: updated.update(run_at=run_at, failed=failed),
    )

    scheduler_module._retry_failed_rebound_tickers(scan_run_at, ["ZURN.SW", "UBSG.SW"])

    assert merged["run_at"] == scan_run_at
    assert merged["rows"] == [{"ticker": "ZURN.SW", "drop_pct": -5.0}]
    assert updated["run_at"] == scan_run_at
    assert updated["failed"] == ["UBSG.SW"]  # still failing, ZURN.SW recovered


def test_retry_failed_rebound_tickers_skips_symbols_no_longer_in_discovery(monkeypatch):
    # A previously-failed symbol that's delisted/renamed and no longer
    # appears in a fresh discover_candidates() call is silently dropped
    # from the retry set rather than retried forever.
    scan_run_at = scheduler_module._now()
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"OTHER.SW": {}})
    filter_calls = []
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic",
        lambda candidates: filter_calls.append(candidates) or ({}, [], False),
    )
    monkeypatch.setattr(scheduler_module.scan_persistence, "merge_rebound_retry_rows", lambda run_at, rows: None)
    updated = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "update_rebound_run_failed_tickers",
        lambda run_at, failed: updated.update(failed=failed),
    )

    scheduler_module._retry_failed_rebound_tickers(scan_run_at, ["DELISTED.SW"])

    assert filter_calls == [{}]  # DELISTED.SW wasn't in discovery, nothing to retry-fetch
    assert updated["failed"] == []


def test_trigger_rebound_retry_starts_a_retry_when_todays_run_has_failures(monkeypatch):
    scheduler_module._rebound_running = False
    today_run_at = scheduler_module._now()
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_rebound_scan",
        lambda: ([{"ticker": "NESN.SW"}], today_run_at, ["ZURN.SW"]),
    )
    started = threading.Event()
    monkeypatch.setattr(
        scheduler_module, "_retry_failed_rebound_tickers",
        lambda run_at, failed: started.set(),
    )

    result = scheduler_module.trigger_rebound_retry()

    assert result is True
    assert started.wait(timeout=2) is True


def test_trigger_rebound_retry_noops_when_todays_run_is_clean(monkeypatch):
    scheduler_module._rebound_running = False
    today_run_at = scheduler_module._now()
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_rebound_scan",
        lambda: ([{"ticker": "NESN.SW"}], today_run_at, []),
    )
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_rebound_retry() is False


def test_trigger_rebound_retry_noops_when_failures_are_from_a_previous_day(monkeypatch):
    scheduler_module._rebound_running = False
    yesterday_run_at = scheduler_module._now() - datetime.timedelta(days=1)
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_rebound_scan",
        lambda: ([{"ticker": "NESN.SW"}], yesterday_run_at, ["ZURN.SW"]),
    )
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_rebound_retry() is False


def test_trigger_rebound_retry_noops_when_nothing_saved_yet(monkeypatch):
    scheduler_module._rebound_running = False
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_rebound_scan", lambda: ([], None, []))
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_rebound_retry() is False


def test_trigger_rebound_retry_noops_when_already_running(monkeypatch):
    scheduler_module._rebound_running = True
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_rebound_retry() is False


def test_retry_failed_indicator_tickers_recomputes_every_threshold(monkeypatch):
    scan_run_at = scheduler_module._now()
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"ZURN.SW": {"q": 1}})
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic",
        lambda candidates: ({"ZURN.SW": {"name": "Zurich"}}, [], False),
    )
    scanned_thresholds = []
    monkeypatch.setattr(
        scheduler_module.swiss_volatility_indicator, "run_scan",
        lambda domestic, threshold_pct: scanned_thresholds.append(threshold_pct) or pd.DataFrame(
            [{"ticker": "ZURN.SW", "loss_days": 3, "gain_days": 2}]
        ),
    )
    merged = []
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "merge_indicator_retry_rows",
        lambda run_at, threshold_pct, rows: merged.append((threshold_pct, rows)),
    )
    updated = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "update_indicator_run_failed_tickers",
        lambda run_at, failed: updated.update(failed=failed),
    )

    scheduler_module._retry_failed_indicator_tickers(scan_run_at, ["ZURN.SW"])

    assert scanned_thresholds == list(ALLOWED_THRESHOLD_PCTS)
    assert [m[0] for m in merged] == list(ALLOWED_THRESHOLD_PCTS)
    assert updated["failed"] == []


def test_trigger_indicator_retry_starts_a_retry_when_this_months_run_has_failures(monkeypatch):
    scheduler_module._indicator_running = False
    this_run_at = scheduler_module._now()
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_indicator_scan",
        lambda threshold_pct: ([{"ticker": "NESN.SW"}], this_run_at, ["ZURN.SW"]),
    )
    started = threading.Event()
    monkeypatch.setattr(
        scheduler_module, "_retry_failed_indicator_tickers",
        lambda run_at, failed: started.set(),
    )

    result = scheduler_module.trigger_indicator_retry()

    assert result is True
    assert started.wait(timeout=2) is True


def test_trigger_indicator_retry_noops_when_this_months_run_is_clean(monkeypatch):
    scheduler_module._indicator_running = False
    this_run_at = scheduler_module._now()
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_indicator_scan",
        lambda threshold_pct: ([{"ticker": "NESN.SW"}], this_run_at, []),
    )
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_indicator_retry() is False


def test_trigger_indicator_retry_noops_when_nothing_saved_yet(monkeypatch):
    scheduler_module._indicator_running = False
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_indicator_scan", lambda threshold_pct: ([], None, []))
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_indicator_retry() is False


def test_trigger_indicator_retry_noops_when_failures_are_from_a_previous_month(monkeypatch):
    scheduler_module._indicator_running = False
    now = scheduler_module._now()
    last_month = now.replace(day=1) - datetime.timedelta(days=1)
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "get_latest_indicator_scan",
        lambda threshold_pct: ([{"ticker": "NESN.SW"}], last_month, ["ZURN.SW"]),
    )
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_indicator_retry() is False


def test_trigger_indicator_retry_noops_when_already_running(monkeypatch):
    scheduler_module._indicator_running = True
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_indicator_retry() is False


# --- Guarded execution + manual trigger (is_*_scan_running() is the one
# flag the frontend reads, for a full scan and a retry alike) ---


def test_is_rebound_scan_running_reflects_module_state():
    scheduler_module._rebound_running = False
    assert scheduler_module.is_rebound_scan_running() is False
    scheduler_module._rebound_running = True
    assert scheduler_module.is_rebound_scan_running() is True


def test_run_rebound_scan_guarded_is_a_noop_when_already_running(monkeypatch):
    scheduler_module._rebound_running = True
    calls = []
    monkeypatch.setattr(scheduler_module, "_run_rebound_scan", lambda: calls.append(1))

    scheduler_module._run_rebound_scan_guarded()

    assert calls == []  # never ran the actual pipeline
    assert scheduler_module.is_rebound_scan_running() is True  # untouched, still whatever it was


def test_run_rebound_scan_guarded_runs_and_clears_flag_on_success(monkeypatch):
    scheduler_module._rebound_running = False
    calls = []
    monkeypatch.setattr(scheduler_module, "_run_rebound_scan", lambda: calls.append(1))

    scheduler_module._run_rebound_scan_guarded()

    assert calls == [1]
    assert scheduler_module.is_rebound_scan_running() is False


def test_run_rebound_scan_guarded_clears_flag_even_on_failure(monkeypatch):
    scheduler_module._rebound_running = False

    def _boom():
        raise RuntimeError("simulated scan failure")

    monkeypatch.setattr(scheduler_module, "_run_rebound_scan", _boom)

    scheduler_module._run_rebound_scan_guarded()  # must not raise - caught and logged

    assert scheduler_module.is_rebound_scan_running() is False


def test_trigger_rebound_scan_returns_false_and_starts_nothing_when_already_running(monkeypatch):
    scheduler_module._rebound_running = True
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_rebound_scan() is False


def test_trigger_rebound_scan_runs_the_guarded_pipeline_in_the_background(monkeypatch):
    scheduler_module._rebound_running = False
    started = threading.Event()
    finish = threading.Event()

    def fake_run():
        started.set()
        finish.wait(timeout=2)

    monkeypatch.setattr(scheduler_module, "_run_rebound_scan", fake_run)

    result = scheduler_module.trigger_rebound_scan()

    assert result is True
    assert started.wait(timeout=2) is True
    assert scheduler_module.is_rebound_scan_running() is True

    finish.set()
    for _ in range(100):
        if not scheduler_module.is_rebound_scan_running():
            break
        time.sleep(0.02)
    assert scheduler_module.is_rebound_scan_running() is False


def test_is_indicator_scan_running_reflects_module_state():
    scheduler_module._indicator_running = False
    assert scheduler_module.is_indicator_scan_running() is False
    scheduler_module._indicator_running = True
    assert scheduler_module.is_indicator_scan_running() is True


def test_run_indicator_scans_guarded_is_a_noop_when_already_running(monkeypatch):
    scheduler_module._indicator_running = True
    calls = []
    monkeypatch.setattr(scheduler_module, "_run_indicator_scans", lambda: calls.append(1))

    scheduler_module._run_indicator_scans_guarded()

    assert calls == []


def test_trigger_indicator_scan_returns_false_when_already_running(monkeypatch):
    scheduler_module._indicator_running = True
    monkeypatch.setattr(
        scheduler_module.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )

    assert scheduler_module.trigger_indicator_scan() is False


def test_trigger_indicator_scan_runs_the_guarded_pipeline_in_the_background(monkeypatch):
    scheduler_module._indicator_running = False
    started = threading.Event()
    finish = threading.Event()

    def fake_run():
        started.set()
        finish.wait(timeout=2)

    monkeypatch.setattr(scheduler_module, "_run_indicator_scans", fake_run)

    result = scheduler_module.trigger_indicator_scan()

    assert result is True
    assert started.wait(timeout=2) is True
    assert scheduler_module.is_indicator_scan_running() is True

    finish.set()
    for _ in range(100):
        if not scheduler_module.is_indicator_scan_running():
            break
        time.sleep(0.02)
    assert scheduler_module.is_indicator_scan_running() is False


def test_retry_failed_rebound_tickers_guarded_is_a_noop_when_already_running(monkeypatch):
    scheduler_module._rebound_running = True
    calls = []
    monkeypatch.setattr(scheduler_module, "_retry_failed_rebound_tickers", lambda run_at, failed: calls.append(1))

    scheduler_module._retry_failed_rebound_tickers_guarded(scheduler_module._now(), ["ZURN.SW"])

    assert calls == []
    assert scheduler_module.is_rebound_scan_running() is True  # untouched


def test_retry_failed_rebound_tickers_guarded_runs_and_clears_flag(monkeypatch):
    scheduler_module._rebound_running = False
    calls = []
    monkeypatch.setattr(scheduler_module, "_retry_failed_rebound_tickers", lambda run_at, failed: calls.append(1))

    scheduler_module._retry_failed_rebound_tickers_guarded(scheduler_module._now(), ["ZURN.SW"])

    assert calls == [1]
    assert scheduler_module.is_rebound_scan_running() is False


def test_retry_failed_indicator_tickers_guarded_is_a_noop_when_already_running(monkeypatch):
    scheduler_module._indicator_running = True
    calls = []
    monkeypatch.setattr(scheduler_module, "_retry_failed_indicator_tickers", lambda run_at, failed: calls.append(1))

    scheduler_module._retry_failed_indicator_tickers_guarded(scheduler_module._now(), ["ZURN.SW"])

    assert calls == []
    assert scheduler_module.is_indicator_scan_running() is True  # untouched


def test_retry_failed_indicator_tickers_guarded_runs_and_clears_flag(monkeypatch):
    scheduler_module._indicator_running = False
    calls = []
    monkeypatch.setattr(scheduler_module, "_retry_failed_indicator_tickers", lambda run_at, failed: calls.append(1))

    scheduler_module._retry_failed_indicator_tickers_guarded(scheduler_module._now(), ["ZURN.SW"])

    assert calls == [1]
    assert scheduler_module.is_indicator_scan_running() is False

import datetime

import pandas as pd

import app.services.scheduler as scheduler_module
from app.services.swiss_volatility_indicator import ALLOWED_THRESHOLD_PCTS


class _FakeAPScheduler:
    """Stands in for apscheduler.schedulers.asyncio.AsyncIOScheduler - real
    one would actually start a background thread/event-loop integration,
    which these tests have no need to exercise (see conftest.py's
    RESEARCH_SCHEDULER_DISABLED guard for why the real one must never run
    during tests anyway)."""

    def __init__(self, *args, **kwargs):
        self.jobs = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_called = True


def _job_ids(fake):
    return [j["id"] for j in fake.jobs]


def _install_fake(monkeypatch):
    monkeypatch.setenv("RESEARCH_SCHEDULER_DISABLED", "0")
    monkeypatch.setattr(scheduler_module, "AsyncIOScheduler", _FakeAPScheduler)
    monkeypatch.setattr(scheduler_module, "_scheduler", None)


def teardown_function(_fn):
    # start()/shutdown() mutate the module-level _scheduler global - reset
    # it after every test in this file regardless of pass/fail, so one
    # test's state can't leak into the next.
    scheduler_module._scheduler = None


# --- RESEARCH_SCHEDULER_DISABLED guard ---


def test_start_noops_when_disabled(monkeypatch):
    monkeypatch.setenv("RESEARCH_SCHEDULER_DISABLED", "1")
    monkeypatch.setattr(
        scheduler_module, "AsyncIOScheduler",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("AsyncIOScheduler should not be constructed")),
    )
    scheduler_module.start()
    assert scheduler_module.is_running() is False


# --- staleness checks ---


def test_is_rebound_stale_true_when_never_run(monkeypatch):
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_rebound_scan", lambda: ([], None))
    assert scheduler_module._is_rebound_stale() is True


def test_is_rebound_stale_false_when_run_today(monkeypatch):
    today = scheduler_module._now()
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_rebound_scan", lambda: ([{}], today))
    assert scheduler_module._is_rebound_stale() is False


def test_is_rebound_stale_true_when_run_yesterday(monkeypatch):
    yesterday = scheduler_module._now() - datetime.timedelta(days=1)
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_rebound_scan", lambda: ([{}], yesterday))
    assert scheduler_module._is_rebound_stale() is True


def test_is_indicator_stale_true_when_never_run(monkeypatch):
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_indicator_scan", lambda threshold_pct: ([], None))
    assert scheduler_module._is_indicator_stale() is True


def test_is_indicator_stale_false_when_run_this_month(monkeypatch):
    this_month = scheduler_module._now()
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_indicator_scan", lambda threshold_pct: ([{}], this_month))
    assert scheduler_module._is_indicator_stale() is False


def test_is_indicator_stale_true_when_run_last_month(monkeypatch):
    now = scheduler_module._now()
    # Roll back to the 1st of the current month, then one more day, to land
    # safely in the previous month regardless of what day "now" actually is.
    last_month = now.replace(day=1) - datetime.timedelta(days=1)
    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_indicator_scan", lambda threshold_pct: ([{}], last_month))
    assert scheduler_module._is_indicator_stale() is True


def test_is_indicator_stale_checks_only_the_first_allowed_threshold(monkeypatch):
    # Documents the deliberate simplification (see _is_indicator_stale's
    # own comment): all three thresholds are always written together by
    # run_indicator_scans_job, so checking ALLOWED_THRESHOLD_PCTS[0] alone
    # is enough - this test pins that assumption instead of leaving it
    # implicit.
    seen_thresholds = []

    def fake_get(threshold_pct):
        seen_thresholds.append(threshold_pct)
        return [{}], scheduler_module._now()

    monkeypatch.setattr(scheduler_module.scan_persistence, "get_latest_indicator_scan", fake_get)
    scheduler_module._is_indicator_stale()
    assert seen_thresholds == [ALLOWED_THRESHOLD_PCTS[0]]


# --- start() catch-up scheduling ---


def test_start_schedules_catchup_jobs_when_both_stale(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr(scheduler_module, "_is_rebound_stale", lambda: True)
    monkeypatch.setattr(scheduler_module, "_is_indicator_stale", lambda: True)

    scheduler_module.start()

    fake = scheduler_module._scheduler
    assert fake.started is True
    assert "rebound_scan" in _job_ids(fake)  # the recurring cron job
    assert "volatility_indicator_scan" in _job_ids(fake)  # the recurring cron job
    assert "rebound_scan_catchup" in _job_ids(fake)
    assert "volatility_indicator_scan_catchup" in _job_ids(fake)


def test_start_skips_catchup_jobs_when_both_fresh(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr(scheduler_module, "_is_rebound_stale", lambda: False)
    monkeypatch.setattr(scheduler_module, "_is_indicator_stale", lambda: False)

    scheduler_module.start()

    fake = scheduler_module._scheduler
    ids = _job_ids(fake)
    assert "rebound_scan_catchup" not in ids
    assert "volatility_indicator_scan_catchup" not in ids
    # The recurring cron jobs are always registered regardless of staleness.
    assert "rebound_scan" in ids
    assert "volatility_indicator_scan" in ids


def test_shutdown_calls_underlying_scheduler_and_clears_reference(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr(scheduler_module, "_is_rebound_stale", lambda: False)
    monkeypatch.setattr(scheduler_module, "_is_indicator_stale", lambda: False)
    scheduler_module.start()
    fake = scheduler_module._scheduler

    scheduler_module.shutdown()

    assert fake.shutdown_called is True
    assert scheduler_module.is_running() is False


# --- run_rebound_scan_job / run_indicator_scans_job (the actual pipeline) ---


def _fake_domestic():
    return {"NESN.SW": {"name": "Nestle"}, "ABBN.SW": {"name": "ABB"}}


def test_run_rebound_scan_job_discovers_scans_and_persists(monkeypatch):
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"NESN.SW": {}, "ABBN.SW": {}})
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic_batched",
        lambda candidates, num_batches, batch_delay_seconds: (_fake_domestic(), []),
    )
    monkeypatch.setattr(
        scheduler_module.swiss_crash_rebound, "run_scan",
        lambda domestic: pd.DataFrame([{"ticker": "NESN.SW", "drop_pct": -5.2}]),
    )
    saved = {}
    monkeypatch.setattr(
        scheduler_module.scan_persistence, "save_rebound_scan",
        lambda rows, scan_run_at=None: saved.update(rows=rows, scan_run_at=scan_run_at),
    )

    scheduler_module.run_rebound_scan_job()

    assert saved["rows"] == [{"ticker": "NESN.SW", "drop_pct": -5.2}]


def test_run_indicator_scans_job_scans_every_allowed_threshold_off_one_discovery(monkeypatch):
    discover_calls = []
    monkeypatch.setattr(scheduler_module.swiss_universe, "discover_candidates", lambda: {"NESN.SW": {}})
    monkeypatch.setattr(
        scheduler_module.swiss_universe, "filter_domestic_batched",
        lambda candidates, num_batches, batch_delay_seconds: discover_calls.append(1) or (_fake_domestic(), []),
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
        lambda rows, threshold_pct, scan_run_at=None: saved.append((threshold_pct, rows, scan_run_at)),
    )

    scheduler_module.run_indicator_scans_job()

    # One discovery pass shared across all thresholds, not one per threshold.
    assert len(discover_calls) == 1
    assert scanned_thresholds == list(ALLOWED_THRESHOLD_PCTS)
    assert [s[0] for s in saved] == list(ALLOWED_THRESHOLD_PCTS)
    # All three saved under the SAME scan_run_at - one shared run, not
    # three independently-timestamped ones.
    assert len({s[2] for s in saved}) == 1

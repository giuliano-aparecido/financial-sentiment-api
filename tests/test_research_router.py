import datetime

import pytest

from app.routers import research as research_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


@pytest.fixture(autouse=True)
def reset_status_rate_limit():
    # The status/result routes' "30/minute" limit is keyed "global" (see
    # app/limiter.py's rate_limit_key), not per-test - and since conftest.py
    # imports `app` once at module scope, every test in this file shares
    # the SAME Limiter/storage instance. Without resetting between tests,
    # an early test's requests could push a later one over budget
    # regardless of what it's actually testing. The /start routes no
    # longer have their own @limiter.limit override (see research.py's
    # own comment for why - single-flight is the real protection now),
    # so they fall back to the app-wide 10/minute default, also covered
    # by this same reset.
    research_router.limiter.reset()
    yield


# --- today (big-loss) scan - unchanged, still live/on-demand ---


def test_today_start_requires_api_key(client):
    response = client.post("/api/research/volatility/today/start")
    assert response.status_code == 401


def test_today_status_requires_api_key(client):
    response = client.get("/api/research/volatility/today/status")
    assert response.status_code == 401


def test_today_start_delegates_to_research_job(client, monkeypatch):
    monkeypatch.setattr(
        research_router, "start_today_scan",
        lambda: {"status": "running", "started_at": "now"},
    )
    response = client.post("/api/research/volatility/today/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "running", "started_at": "now"}


def test_today_status_delegates_to_research_job(client, monkeypatch):
    fake_status = {"status": "done", "today_screener": [{"ticker": "NVDA.SW"}]}
    monkeypatch.setattr(research_router, "get_today_status", lambda: fake_status)
    response = client.get("/api/research/volatility/today/status", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == fake_status


# --- rebound: scheduled-scan read + manual trigger ---


def test_get_rebound_scan_requires_api_key(client):
    response = client.get("/api/research/volatility/rebound")
    assert response.status_code == 401


def test_get_rebound_scan_reads_from_persistence(client, monkeypatch):
    monkeypatch.setattr(
        research_router.scan_persistence, "get_latest_rebound_scan",
        lambda: ([{"ticker": "NESN.SW"}], datetime.datetime(2026, 8, 19, 6, 0, tzinfo=datetime.timezone.utc)),
    )
    monkeypatch.setattr(research_router.scheduler, "is_rebound_scan_running", lambda: False)
    response = client.get("/api/research/volatility/rebound", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == [{"ticker": "NESN.SW"}]
    assert body["scan_run_at"] == "2026-08-19T06:00:00+00:00"
    assert body["is_running"] is False


def test_get_rebound_scan_returns_null_scan_run_at_when_nothing_saved(client, monkeypatch):
    monkeypatch.setattr(research_router.scan_persistence, "get_latest_rebound_scan", lambda: ([], None))
    monkeypatch.setattr(research_router.scheduler, "is_rebound_scan_running", lambda: False)
    response = client.get("/api/research/volatility/rebound", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"rows": [], "scan_run_at": None, "is_running": False}


def test_get_rebound_scan_reports_is_running_true_regardless_of_trigger_source(client, monkeypatch):
    # Whether a scan currently running was started by the cron or by a
    # manual Refresh click, scheduler.is_rebound_scan_running() is the
    # ONE source of truth this route reads - see scheduler.py's own
    # module docstring for why that unification matters.
    monkeypatch.setattr(research_router.scan_persistence, "get_latest_rebound_scan", lambda: ([], None))
    monkeypatch.setattr(research_router.scheduler, "is_rebound_scan_running", lambda: True)
    response = client.get("/api/research/volatility/rebound", headers={"X-API-Key": VALID_KEY})
    assert response.json()["is_running"] is True


def test_rebound_start_requires_api_key(client):
    response = client.post("/api/research/volatility/rebound/start")
    assert response.status_code == 401


def test_rebound_start_triggers_the_scheduler_and_returns_started_true(client, monkeypatch):
    monkeypatch.setattr(research_router.scheduler, "trigger_rebound_scan", lambda: True)
    response = client.post("/api/research/volatility/rebound/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"started": True, "is_running": True}


def test_rebound_start_returns_started_false_when_already_running(client, monkeypatch):
    # Single-flight, matching the pre-scheduling behavior: clicking
    # Refresh while a scan is already in progress is a safe no-op, not
    # an error - see scheduler.trigger_rebound_scan's own docstring.
    monkeypatch.setattr(research_router.scheduler, "trigger_rebound_scan", lambda: False)
    response = client.post("/api/research/volatility/rebound/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"started": False, "is_running": True}


# --- volatility-indicator: scheduled-scan read + manual trigger ---


def test_get_indicator_scan_requires_api_key(client):
    response = client.get("/api/research/volatility/indicator?threshold_pct=2.0")
    assert response.status_code == 401


def test_get_indicator_scan_requires_threshold_pct_query_param(client):
    response = client.get("/api/research/volatility/indicator", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 422


def test_get_indicator_scan_rejects_a_value_outside_the_allowed_set(client):
    response = client.get(
        "/api/research/volatility/indicator?threshold_pct=4.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 422


def test_get_indicator_scan_reads_from_persistence_scoped_to_threshold(client, monkeypatch):
    calls = []

    def fake_get(threshold_pct):
        calls.append(threshold_pct)
        return [{"ticker": "NESN.SW"}], None

    monkeypatch.setattr(research_router.scan_persistence, "get_latest_indicator_scan", fake_get)
    monkeypatch.setattr(research_router.scheduler, "is_indicator_scan_running", lambda: False)
    response = client.get(
        "/api/research/volatility/indicator?threshold_pct=5.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert calls == [5.0]
    body = response.json()
    assert body["rows"] == [{"ticker": "NESN.SW"}]
    assert body["is_running"] is False


def test_get_indicator_scan_is_running_is_the_same_regardless_of_selected_threshold(client, monkeypatch):
    # Only ONE indicator scan runs at a time, covering all three
    # thresholds together (see scheduler._run_indicator_scans) - there's
    # no per-threshold running state.
    monkeypatch.setattr(research_router.scan_persistence, "get_latest_indicator_scan", lambda threshold_pct: ([], None))
    monkeypatch.setattr(research_router.scheduler, "is_indicator_scan_running", lambda: True)
    response = client.get(
        "/api/research/volatility/indicator?threshold_pct=2.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.json()["is_running"] is True


def test_indicator_start_requires_api_key(client):
    response = client.post("/api/research/volatility/indicator/start?threshold_pct=2.0")
    assert response.status_code == 401


def test_indicator_start_requires_threshold_pct_query_param(client):
    response = client.post("/api/research/volatility/indicator/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 422


def test_indicator_start_rejects_a_value_outside_the_allowed_set(client):
    response = client.post(
        "/api/research/volatility/indicator/start?threshold_pct=4.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 422


def test_indicator_start_accepts_each_allowed_threshold(client, monkeypatch):
    monkeypatch.setattr(research_router.scheduler, "trigger_indicator_scan", lambda: True)
    for threshold in (2.0, 3.0, 5.0):
        response = client.post(
            f"/api/research/volatility/indicator/start?threshold_pct={threshold}", headers={"X-API-Key": VALID_KEY},
        )
        assert response.status_code == 200
        assert response.json() == {"started": True, "is_running": True}


def test_indicator_start_triggers_the_scheduler_regardless_of_which_threshold_was_selected(client, monkeypatch):
    # trigger_indicator_scan() always scans every threshold (see its own
    # docstring) - the selected threshold_pct is validated but otherwise
    # unused by the trigger call itself.
    calls = []
    monkeypatch.setattr(research_router.scheduler, "trigger_indicator_scan", lambda: calls.append(1) or True)
    response = client.post(
        "/api/research/volatility/indicator/start?threshold_pct=3.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert calls == [1]


def test_indicator_start_returns_started_false_when_already_running(client, monkeypatch):
    monkeypatch.setattr(research_router.scheduler, "trigger_indicator_scan", lambda: False)
    response = client.post(
        "/api/research/volatility/indicator/start?threshold_pct=2.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json() == {"started": False, "is_running": True}


def test_start_routes_allow_rapid_repeated_calls_without_a_business_cooldown(client, monkeypatch):
    # Regression guard: /start routes used to have their own stricter
    # "1/5minutes" override, which blocked a genuinely new request (e.g.
    # a different threshold, or simply trying again after a scan
    # finished) with a misleading "already started recently" error - see
    # research.py's own comment for why this was removed in favor of
    # scheduler.py's single-flight-per-table protection.
    monkeypatch.setattr(research_router.scheduler, "trigger_indicator_scan", lambda: True)
    for _ in range(3):
        response = client.post(
            "/api/research/volatility/indicator/start?threshold_pct=2.0", headers={"X-API-Key": VALID_KEY},
        )
        assert response.status_code == 200


def test_rebound_and_indicator_start_routes_are_independent(client, monkeypatch):
    rebound_calls = []
    indicator_calls = []
    monkeypatch.setattr(research_router.scheduler, "trigger_rebound_scan", lambda: rebound_calls.append(True) or True)
    monkeypatch.setattr(research_router.scheduler, "trigger_indicator_scan", lambda: indicator_calls.append(True) or True)
    client.post("/api/research/volatility/rebound/start", headers={"X-API-Key": VALID_KEY})
    assert rebound_calls == [True]
    assert indicator_calls == []

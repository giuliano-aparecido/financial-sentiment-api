import pytest

from app.routers import research as research_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


@pytest.fixture(autouse=True)
def reset_status_rate_limit():
    # The status routes' "30/minute" limit is keyed "global" (see
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


def test_rebound_start_requires_api_key(client):
    response = client.post("/api/research/volatility/rebound/start")
    assert response.status_code == 401


def test_rebound_status_requires_api_key(client):
    response = client.get("/api/research/volatility/rebound/status")
    assert response.status_code == 401


def test_rebound_start_delegates_to_research_job(client, monkeypatch):
    monkeypatch.setattr(
        research_router, "start_rebound_scan",
        lambda: {"status": "running", "started_at": "now"},
    )
    response = client.post("/api/research/volatility/rebound/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "running", "started_at": "now"}


def test_rebound_status_delegates_to_research_job(client, monkeypatch):
    fake_status = {"status": "done", "crash_rebound": [{"ticker": "NVDA.SW"}]}
    monkeypatch.setattr(research_router, "get_rebound_status", lambda: fake_status)
    response = client.get("/api/research/volatility/rebound/status", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == fake_status


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


def test_rebound_and_today_start_routes_are_independent(client, monkeypatch):
    # Regression guard: these used to be ONE combined /start route/button
    # - confirms calling one doesn't touch the other's delegate function.
    rebound_calls = []
    today_calls = []
    monkeypatch.setattr(
        research_router, "start_rebound_scan",
        lambda: rebound_calls.append(True) or {"status": "running"},
    )
    monkeypatch.setattr(
        research_router, "start_today_scan",
        lambda: today_calls.append(True) or {"status": "running"},
    )
    client.post("/api/research/volatility/rebound/start", headers={"X-API-Key": VALID_KEY})
    assert rebound_calls == [True]
    assert today_calls == []


# --- volatility-indicator scan (entirely separate endpoints) ---


def test_indicator_start_requires_api_key(client):
    response = client.post("/api/research/volatility/indicator/start?threshold_pct=2.0")
    assert response.status_code == 401


def test_indicator_status_requires_api_key(client):
    response = client.get("/api/research/volatility/indicator/status")
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
    monkeypatch.setattr(
        research_router, "start_indicator_scan",
        lambda threshold_pct: {"status": "running", "threshold_pct": threshold_pct},
    )
    for threshold in (2.0, 3.0, 5.0):
        response = client.post(
            f"/api/research/volatility/indicator/start?threshold_pct={threshold}", headers={"X-API-Key": VALID_KEY},
        )
        assert response.status_code == 200
        assert response.json()["threshold_pct"] == threshold


def test_indicator_start_delegates_to_research_job(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        research_router, "start_indicator_scan",
        lambda threshold_pct: calls.append(threshold_pct) or {"status": "running", "threshold_pct": threshold_pct},
    )
    response = client.post(
        "/api/research/volatility/indicator/start?threshold_pct=3.0", headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert calls == [3.0]
    assert response.json() == {"status": "running", "threshold_pct": 3.0}


def test_indicator_status_delegates_to_research_job(client, monkeypatch):
    fake_status = {"status": "done", "threshold_pct": 2.0, "volatility_indicator": [{"ticker": "NVDA.SW"}]}
    monkeypatch.setattr(research_router, "get_indicator_status", lambda: fake_status)
    response = client.get("/api/research/volatility/indicator/status", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == fake_status


def test_start_routes_allow_rapid_repeated_calls_without_a_business_cooldown(client, monkeypatch):
    # Regression guard: /start routes used to have their own stricter
    # "1/5minutes" override, which blocked a genuinely new request (e.g.
    # a different threshold, or simply trying again after a scan
    # finished) with a misleading "already started recently" error - see
    # research.py's own comment for why this was removed in favor of
    # research_job.py's single-flight-per-scan-type protection.
    monkeypatch.setattr(research_router, "start_indicator_scan", lambda threshold_pct: {"status": "running"})
    for _ in range(3):
        response = client.post(
            "/api/research/volatility/indicator/start?threshold_pct=2.0", headers={"X-API-Key": VALID_KEY},
        )
        assert response.status_code == 200

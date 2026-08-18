import pytest

from app.routers import research as research_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


@pytest.fixture(autouse=True)
def reset_start_scan_rate_limit():
    # The /start route's "1/5minutes" limit is keyed "global" (see
    # app/limiter.py's rate_limit_key), not per-test - and since conftest.py
    # imports `app` once at module scope, every test in this file shares
    # the SAME Limiter/storage instance. Without resetting between tests,
    # only the first test in the file that POSTs to /start would ever see
    # a 200; every later one would get a stale 429 regardless of what it's
    # actually testing.
    research_router.limiter.reset()
    yield


def test_start_scan_requires_api_key(client):
    response = client.post("/api/research/volatility/start")
    assert response.status_code == 401


def test_status_requires_api_key(client):
    response = client.get("/api/research/volatility/status")
    assert response.status_code == 401


def test_start_scan_delegates_to_research_job(client, monkeypatch):
    monkeypatch.setattr(
        research_router, "start_scan",
        lambda: {"status": "running", "started_at": "now"},
    )
    response = client.post("/api/research/volatility/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "running", "started_at": "now"}


def test_start_scan_takes_no_query_params(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        research_router, "start_scan",
        lambda: calls.append(True) or {"status": "running", "started_at": "now"},
    )
    response = client.post("/api/research/volatility/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert calls == [True]


def test_status_delegates_to_research_job(client, monkeypatch):
    fake_status = {
        "status": "done",
        "crash_rebound": [{"ticker": "NVDA.SW"}],
        "today_screener": [],
    }
    monkeypatch.setattr(research_router, "get_status", lambda: fake_status)
    response = client.get("/api/research/volatility/status", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == fake_status


def test_status_idle_before_any_scan_started(client, monkeypatch):
    monkeypatch.setattr(research_router, "get_status", lambda: {"status": "idle"})
    response = client.get("/api/research/volatility/status", headers={"X-API-Key": VALID_KEY})
    assert response.json() == {"status": "idle"}


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
        research_router.limiter.reset()  # each iteration is its own "first call this window"
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

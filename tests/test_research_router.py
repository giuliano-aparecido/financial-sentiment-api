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
        lambda all_caps=False: {"status": "running", "started_at": "now", "all_caps": all_caps},
    )
    response = client.post("/api/research/volatility/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "running", "started_at": "now", "all_caps": False}


def test_start_scan_passes_all_caps_query_param_through(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        research_router, "start_scan",
        lambda all_caps=False: calls.append(all_caps) or {"status": "running", "started_at": "now", "all_caps": all_caps},
    )
    response = client.post("/api/research/volatility/start?all_caps=true", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert calls == [True]
    assert response.json()["all_caps"] is True


def test_start_scan_all_caps_defaults_false_when_omitted(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        research_router, "start_scan",
        lambda all_caps=False: calls.append(all_caps) or {"status": "running", "started_at": "now", "all_caps": all_caps},
    )
    response = client.post("/api/research/volatility/start", headers={"X-API-Key": VALID_KEY})
    assert calls == [False]


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

from app.routers import research as research_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


def test_start_scan_requires_api_key(client):
    response = client.post("/api/research/small-caps/start")
    assert response.status_code == 401


def test_status_requires_api_key(client):
    response = client.get("/api/research/small-caps/status")
    assert response.status_code == 401


def test_start_scan_delegates_to_research_job(client, monkeypatch):
    monkeypatch.setattr(research_router, "start_scan", lambda: {"status": "running", "started_at": "now"})
    response = client.post("/api/research/small-caps/start", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == {"status": "running", "started_at": "now"}


def test_status_delegates_to_research_job(client, monkeypatch):
    fake_status = {
        "status": "done",
        "crash_rebound": [{"ticker": "NVDA.SW"}],
        "today_screener": [],
    }
    monkeypatch.setattr(research_router, "get_status", lambda: fake_status)
    response = client.get("/api/research/small-caps/status", headers={"X-API-Key": VALID_KEY})
    assert response.status_code == 200
    assert response.json() == fake_status


def test_status_idle_before_any_scan_started(client, monkeypatch):
    monkeypatch.setattr(research_router, "get_status", lambda: {"status": "idle"})
    response = client.get("/api/research/small-caps/status", headers={"X-API-Key": VALID_KEY})
    assert response.json() == {"status": "idle"}

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


def test_health_requires_no_auth():
    response = client.get("/health")
    assert response.status_code == 200


def test_analyze_rejects_missing_key():
    response = client.post("/api/analyze", json={"user_query": "AAPL"})
    assert response.status_code == 401


def test_analyze_rejects_wrong_key():
    response = client.post(
        "/api/analyze",
        json={"user_query": "AAPL"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_update_inference_url_rejects_missing_key():
    response = client.post("/api/update-inference-url", json={"url": "https://abc123.ngrok-free.app"})
    assert response.status_code == 401


def test_update_inference_url_rejects_disallowed_host():
    response = client.post(
        "/api/update-inference-url",
        json={"url": "http://169.254.169.254/latest/meta-data"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 400


def test_update_inference_url_accepts_allowed_host():
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://abc123.ngrok-free.app/"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json()["hf_inference_url"] == "https://abc123.ngrok-free.app"


def test_update_inference_url_accepts_ngrok_free_dev_with_path():
    # Regression: ngrok-free.dev is a real, current ngrok free-tier domain
    # (alongside ngrok-free.app) - a Colab tunnel on it was rejected before
    # this suffix was added, and a path like /generate must survive intact
    # since it's part of where analyze_stock actually POSTs.
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://unflawed-washboard-crux.ngrok-free.dev/generate"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json()["hf_inference_url"] == "https://unflawed-washboard-crux.ngrok-free.dev/generate"

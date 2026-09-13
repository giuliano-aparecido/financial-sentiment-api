from app.routers import admin as admin_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


def test_health_requires_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_analyze_rejects_missing_key(client):
    response = client.post("/api/analyze", json={"user_query": "AAPL"})
    assert response.status_code == 401


def test_analyze_rejects_wrong_key(client):
    response = client.post(
        "/api/analyze",
        json={"user_query": "AAPL"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_update_inference_url_rejects_missing_key(client):
    response = client.post("/api/update-inference-url", json={"url": "https://abc123.ngrok-free.app"})
    assert response.status_code == 401


def test_update_inference_url_rejects_disallowed_host(client):
    response = client.post(
        "/api/update-inference-url",
        json={"url": "http://169.254.169.254/latest/meta-data"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 400


def test_update_inference_url_rejects_http_scheme_for_allowed_host(client):
    # The only other rejection test (rejects_disallowed_host) uses a URL
    # that's both http AND a disallowed host, so it never isolates the
    # scheme check on its own - this confirms the https-only guard rejects
    # an otherwise-allowlisted host too, not just non-allowlisted ones.
    response = client.post(
        "/api/update-inference-url",
        json={"url": "http://abc123.ngrok-free.app"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 400


def test_update_inference_url_accepts_allowed_host(client):
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://abc123.ngrok-free.app/"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json()["hf_inference_url"] == "https://abc123.ngrok-free.app"


def test_update_inference_url_rejects_lookalike_host(client):
    # Regression: host.endswith("huggingface.co") used to accept anything
    # ending in that string, not just that domain or a subdomain of it - a
    # registered "evil-huggingface.co" would have passed and then received
    # the HF bearer token on every /api/analyze call.
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://evil-huggingface.co/steal"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 400


def test_update_inference_url_accepts_subdomain_of_allowed_host(client):
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://my-space.huggingface.co/generate"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json()["hf_inference_url"] == "https://my-space.huggingface.co/generate"


def test_update_inference_url_accepts_ngrok_free_dev_with_path(client):
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


def test_update_inference_url_accepts_modal_run(client):
    # modal.run backs the scale-to-zero serving alternative to the Colab/
    # ngrok tunnel (see financial-sentiment-model's modal/serve_model.py) -
    # its endpoint URLs are subdomains like <workspace>--<app>-<fn>.modal.run.
    response = client.post(
        "/api/update-inference-url",
        json={"url": "https://gaparecido--financial-sentiment-reasoner-generate.modal.run"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200
    assert response.json()["hf_inference_url"] == "https://gaparecido--financial-sentiment-reasoner-generate.modal.run"


def test_update_yf_crumb_rejects_missing_key(client):
    response = client.post("/api/update-yf-crumb", json={"crumb": "abc", "cookies": {}})
    assert response.status_code == 401


def test_update_yf_crumb_rejects_wrong_key(client):
    response = client.post(
        "/api/update-yf-crumb",
        json={"crumb": "abc", "cookies": {}},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_update_yf_crumb_calls_reseed_with_request_body(client, monkeypatch):
    calls = []
    monkeypatch.setattr(admin_router, "reseed_yf_session", lambda crumb, cookies: calls.append((crumb, cookies)))

    response = client.post(
        "/api/update-yf-crumb",
        json={"crumb": "fresh-crumb-123", "cookies": {"A1": "abc", "A3": "def"}},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert calls == [("fresh-crumb-123", {"A1": "abc", "A3": "def"})]

def test_get_health_returns_status_online(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_get_health_lists_default_model_and_registered_model_names(client, monkeypatch):
    from app import config
    from app.services import inference

    monkeypatch.setattr(inference, "_inference_urls", {"kim": "https://kim.modal.run", config.DEFAULT_MODEL: "https://x.modal.run"})
    body = client.get("/health").json()
    assert body["default_model"] == config.DEFAULT_MODEL
    assert body["models"] == sorted(["kim", config.DEFAULT_MODEL])
    assert "modal.run" not in str(body)

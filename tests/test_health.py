def test_get_health_returns_status_online(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_head_health_returns_200_not_405(client):
    # Regression guard: confirmed live UptimeRobot's keep-warm ping uses
    # HEAD, and a route declared with only @router.get 405s on it in this
    # FastAPI/Starlette version - see health.py's own comment.
    response = client.head("/health")
    assert response.status_code == 200
    assert response.content == b""

def test_get_health_returns_status_online(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

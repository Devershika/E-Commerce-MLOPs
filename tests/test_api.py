from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_predict_success():
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "forecast_horizon_days": 7,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == "S001"
    assert data["product_id"] == "P0001"
    assert len(data["forecast"]) == 7
    assert "predicted_demand" in data["forecast"][0]


def test_api_predict_invalid_horizon():
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "forecast_horizon_days": -5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

"""
Unit & integration tests for the Demand Forecast API.
Covers:
  - Schema validation (valid / invalid payloads → 200 / 422)
  - /health endpoint
  - /predict with mock predictor
  - /predict with a real model bundle stub
  - Inference module directly
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.inference import run_inference
from src.api.loader import ModelBundle
from src.api.main import app
from src.api.mock_service import mock_predict
from src.api.schemas import PredictRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_payload(**overrides: Any) -> dict:
    base = {
        "store_id": "STORE_01",
        "product_id": "PROD_42",
        "category": "Electronics",
        "forecast_horizon_days": 7,
    }
    base.update(overrides)
    return base


def _make_stub_bundle(horizon_size: int = 7) -> ModelBundle:
    """Build a ModelBundle whose underlying model returns dummy floats."""
    mock_model = MagicMock()
    mock_model.forecast.return_value = [float(10 + i) for i in range(horizon_size)]
    return ModelBundle(model=mock_model, validation_mape=0.12)


# ---------------------------------------------------------------------------
# TestClient fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    """TestClient with no model loaded (mock mode)."""
    with TestClient(app) as c:
        app.state.model_bundle = None
        yield c


@pytest.fixture()
def client_with_model():
    """TestClient with a stub model bundle in state."""
    with TestClient(app) as c:
        app.state.model_bundle = _make_stub_bundle()
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client: TestClient):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "version" in data
        assert data["model_loaded"] is False  # no real model in mock mode

    def test_health_model_loaded_true_when_bundle_set(self, client_with_model: TestClient):
        data = client_with_model.get("/health").json()
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict — payload validation
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_valid_payload_returns_200(self, client: TestClient):
        resp = client.post("/predict", json=_valid_payload())
        assert resp.status_code == 200

    def test_missing_store_id_returns_422(self, client: TestClient):
        payload = _valid_payload()
        del payload["store_id"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_product_id_returns_422(self, client: TestClient):
        payload = _valid_payload()
        del payload["product_id"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_horizon_zero_returns_422(self, client: TestClient):
        assert client.post("/predict", json=_valid_payload(forecast_horizon_days=0)).status_code == 422

    def test_horizon_negative_returns_422(self, client: TestClient):
        assert client.post("/predict", json=_valid_payload(forecast_horizon_days=-1)).status_code == 422

    def test_horizon_over_90_returns_422(self, client: TestClient):
        assert client.post("/predict", json=_valid_payload(forecast_horizon_days=91)).status_code == 422

    def test_blank_store_id_returns_422(self, client: TestClient):
        assert client.post("/predict", json=_valid_payload(store_id="   ")).status_code == 422

    def test_blank_product_id_returns_422(self, client: TestClient):
        assert client.post("/predict", json=_valid_payload(product_id="")).status_code == 422


# ---------------------------------------------------------------------------
# /predict — mock mode response shape
# ---------------------------------------------------------------------------
class TestPredictMockResponse:
    def test_response_contains_forecast_list(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload(forecast_horizon_days=5)).json()
        assert isinstance(data["forecast"], list)
        assert len(data["forecast"]) == 5

    def test_forecast_items_have_correct_keys(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload(forecast_horizon_days=3)).json()
        for item in data["forecast"]:
            # DailyForecast serialises forecast_date with alias "date"
            assert "date" in item or "forecast_date" in item
            assert "predicted_demand" in item

    def test_forecast_demand_is_non_negative(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload(forecast_horizon_days=14)).json()
        for item in data["forecast"]:
            assert item["predicted_demand"] >= 0.0

    def test_forecast_dates_are_sequential(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload(forecast_horizon_days=7)).json()
        date_key = "date" if "date" in data["forecast"][0] else "forecast_date"
        dates = [datetime.date.fromisoformat(item[date_key]) for item in data["forecast"]]
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i - 1]).days == 1

    def test_store_and_product_echoed_in_response(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload()).json()
        assert data["store_id"] == "STORE_01"
        assert data["product_id"] == "PROD_42"

    def test_model_name_present(self, client: TestClient):
        data = client.post("/predict", json=_valid_payload()).json()
        assert isinstance(data["model_name"], str)
        assert len(data["model_name"]) > 0


# ---------------------------------------------------------------------------
# /predict — real model bundle path
# ---------------------------------------------------------------------------
class TestPredictWithRealModel:
    def test_returns_200_with_model_bundle(self, client_with_model: TestClient):
        resp = client_with_model.post("/predict", json=_valid_payload(forecast_horizon_days=7))
        assert resp.status_code == 200

    def test_forecast_length_matches_horizon(self, client_with_model: TestClient):
        data = client_with_model.post(
            "/predict", json=_valid_payload(forecast_horizon_days=7)
        ).json()
        assert len(data["forecast"]) == 7

    def test_validation_mape_present(self, client_with_model: TestClient):
        data = client_with_model.post("/predict", json=_valid_payload()).json()
        assert data["validation_mape"] == pytest.approx(0.12)

    def test_forecast_values_are_numeric(self, client_with_model: TestClient):
        data = client_with_model.post(
            "/predict", json=_valid_payload(forecast_horizon_days=7)
        ).json()
        for item in data["forecast"]:
            assert isinstance(item["predicted_demand"], (int, float))


# ---------------------------------------------------------------------------
# Inference module (unit tests, no HTTP)
# ---------------------------------------------------------------------------
class TestRunInferenceDirect:
    def _make_request(self, horizon: int = 5) -> PredictRequest:
        return PredictRequest(
            store_id="S001",
            product_id="P001",
            category="Electronics",
            forecast_horizon_days=horizon,
        )

    def test_mock_returns_correct_length(self):
        req = self._make_request(horizon=10)
        resp = run_inference(req, model_bundle=None)
        assert len(resp.forecast) == 10

    def test_mock_dates_start_from_today(self):
        req = self._make_request(horizon=3)
        resp = run_inference(req, model_bundle=None)
        today = datetime.date.today()
        assert resp.forecast[0].forecast_date == today

    def test_real_bundle_used_when_present(self):
        bundle = _make_stub_bundle(horizon_size=5)
        req = self._make_request(horizon=5)
        resp = run_inference(req, model_bundle=bundle)
        assert resp.model_name == "Holt-Winters (Fine-Tuned)"
        assert len(resp.forecast) == 5

    def test_fallback_to_mock_on_model_error(self):
        """If the model raises, inference should fall back to mock gracefully."""
        broken_model = MagicMock()
        broken_model.forecast.side_effect = RuntimeError("model broken")
        bundle = ModelBundle(model=broken_model)
        req = self._make_request(horizon=4)
        resp = run_inference(req, model_bundle=bundle)
        # Fell back to mock — still returns a valid response
        assert len(resp.forecast) == 4
        assert "Mock" in resp.model_name


# ---------------------------------------------------------------------------
# Mock predictor (unit tests)
# ---------------------------------------------------------------------------
class TestMockPredict:
    def test_returns_correct_count(self):
        results = mock_predict("S1", "P1", "Electronics", 7)
        assert len(results) == 7

    def test_first_date_is_today(self):
        results = mock_predict("S1", "P1", "Electronics", 3)
        assert results[0].forecast_date == datetime.date.today()

    def test_all_demands_non_negative(self):
        results = mock_predict("S1", "P1", "Electronics", 30)
        for item in results:
            assert item.predicted_demand >= 0.0

    def test_horizon_one(self):
        results = mock_predict("S1", "P1", "Electronics", 1)
        assert len(results) == 1

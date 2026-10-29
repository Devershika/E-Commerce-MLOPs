"""
End-to-end inference logic for the demand forecasting API.

Dispatches to either:
  - The real Holt-Winters model bundle (when loaded), or
  - The mock predictor (development / CI fallback).
"""
from __future__ import annotations

import datetime
import logging
from typing import List, Optional

from src.api.loader import ModelBundle
from src.api.mock_service import mock_predict
from src.api.schemas import DailyForecast, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

_MOCK_MODEL_NAME = "MockForecaster (dev)"
_MOCK_MAPE = None


def run_inference(
    request: PredictRequest,
    model_bundle: Optional[ModelBundle],
) -> PredictResponse:
    """
    Generate a demand forecast for the given request.

    Args:
        request:      Validated PredictRequest payload.
        model_bundle: Loaded ModelBundle, or None to use mock predictor.

    Returns:
        PredictResponse with a list of daily forecasts.
    """
    horizon = request.forecast_horizon_days

    if model_bundle is not None:
        # --- Real model path ---
        logger.info(
            "Running inference with %s for store=%s product=%s horizon=%d",
            model_bundle.model_name,
            request.store_id,
            request.product_id,
            horizon,
        )
        try:
            raw_forecasts: list[float] = model_bundle.forecast(horizon)
            forecast_items = _build_forecast_items(raw_forecasts)
            return PredictResponse(
                store_id=request.store_id,
                product_id=request.product_id,
                forecast=forecast_items,
                model_name=model_bundle.model_name,
                validation_mape=model_bundle.validation_mape,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Real model inference failed (%s); falling back to mock: %s",
                model_bundle.model_name,
                exc,
            )
            # Fall through to mock on failure

    # --- Mock / fallback path ---
    logger.info(
        "Using mock predictor for store=%s product=%s horizon=%d",
        request.store_id,
        request.product_id,
        horizon,
    )
    mock_items = mock_predict(
        store_id=request.store_id,
        product_id=request.product_id,
        category=request.category,
        horizon=horizon,
    )
    return PredictResponse(
        store_id=request.store_id,
        product_id=request.product_id,
        forecast=mock_items,
        model_name=_MOCK_MODEL_NAME,
        validation_mape=_MOCK_MAPE,
    )


def _build_forecast_items(raw_forecasts: List[float]) -> List[DailyForecast]:
    """
    Convert a list of raw float predictions into DailyForecast objects.

    Demand values are clipped to zero to avoid negative predictions.
    """
    today = datetime.date.today()
    return [
        DailyForecast(
            forecast_date=today + datetime.timedelta(days=i),
            predicted_demand=round(max(0.0, val), 2),
        )
        for i, val in enumerate(raw_forecasts)
    ]

"""
Mock forecaster used while the real Holt-Winters model artifact is not yet available.
Swap this out in loader.py once models/holt_winters_model.joblib is present.
"""
from __future__ import annotations

import datetime
from typing import List

from src.api.schemas import DailyForecast


def mock_predict(
    store_id: str,
    product_id: str,
    category: str,
    horizon: int,
) -> List[DailyForecast]:
    """
    Return deterministic dummy forecast values.

    The formula produces a gentle upward ramp so responses look
    plausible in a UI without any model loaded.
    """
    today = datetime.date.today()
    return [
        DailyForecast(
            forecast_date=today + datetime.timedelta(days=i),
            predicted_demand=round(50.0 + i * 1.5, 2),
        )
        for i in range(horizon)
    ]

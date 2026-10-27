import numpy as np
import pandas as pd

from demand_forecast_engine.models.holt_winters import HoltWintersForecaster


def test_holt_winters_forecast_returns_expected_length():
    ts = pd.Series(
        [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0,
         180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0,
         260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0,
         340.0, 350.0, 360.0, 370.0, 380.0, 390.0],
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
        name="demand",
    )

    model = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=7)
    fitted = model.fit(ts)

    forecast = fitted.forecast(5)

    assert len(forecast) == 5
    assert np.all(np.isfinite(forecast.to_numpy()))
    assert forecast.name == "demand"

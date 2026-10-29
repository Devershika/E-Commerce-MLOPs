import pandas as pd
from apps.app import fetch_forecast


def test_fetch_forecast_mock_fallback():
    # Calling with invalid backend url triggers the simulated fallback cleanly
    df_fc, model_name, is_mock = fetch_forecast(
        api_url="http://invalid-endpoint:9999",
        store_id="S001",
        product_id="P0001",
        category="Electronics",
        horizon_days=7,
        last_date=pd.to_datetime("2024-01-01").date(),
    )
    assert is_mock is True
    assert len(df_fc) == 7
    assert "predicted_demand" in df_fc.columns
    assert "date" in df_fc.columns
    assert (df_fc["predicted_demand"] > 0).all()

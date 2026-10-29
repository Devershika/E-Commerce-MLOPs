from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersForecaster:
    """Production-ready Holt-Winters forecaster with optional log transform."""

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 7,
        damped_trend: bool = False,
        use_log: bool = True,
        initialization_method: str = "estimated",
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_log = use_log
        self.initialization_method = initialization_method
        self.model_ = None
        self.series_ = None

    def fit(self, series: pd.Series):
        """Fit the model to the provided univariate time series."""
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        series = series.copy().sort_index()
        if series.empty:
            raise ValueError("Input series cannot be empty.")

        if not pd.api.types.is_datetime64_any_dtype(series.index):
            series.index = pd.to_datetime(series.index)

        series = series.dropna()
        if series.empty:
            raise ValueError("Input series contains no non-null values.")

        if self.use_log:
            transformed = np.log1p(series.astype(float))
        else:
            transformed = series.astype(float)

        self.series_ = series
        self._transformed_series_ = transformed
        self.model_ = ExponentialSmoothing(
            transformed,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method=self.initialization_method,
        ).fit(optimized=True)
        return self

    def forecast(self, horizon: int) -> pd.Series:
        """Return predictions for the next `horizon` periods."""
        if self.model_ is None:
            raise ValueError("Model is not fitted. Call fit() before forecast().")

        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")

        forecast_values = self.model_.forecast(horizon)

        if self.use_log:
            forecast_values = np.expm1(forecast_values)

        if isinstance(forecast_values, pd.Series):
            predictions = forecast_values
        else:
            predictions = pd.Series(forecast_values)

        predictions = predictions.rename(self.series_.name or "demand")

        if hasattr(self.series_, "index") and len(self.series_.index) > 0:
            last_index = self.series_.index[-1]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=Warning, message=".*infer_freq.*")
                inferred = pd.infer_freq(self.series_.index)
            freq = inferred if inferred else "D"
            future_index = pd.date_range(
                start=last_index + pd.Timedelta(days=1), periods=horizon, freq=freq
            )
            predictions.index = future_index

        return predictions

    def predict(self, horizon: int) -> pd.Series:
        return self.forecast(horizon)

    def save(self, path: str, metadata: Optional[dict] = None) -> str:
        import joblib

        bundle = {
            "model": self.model_,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "use_log": self.use_log,
            "series_name": self.series_.name if self.series_ is not None else "demand",
            "metadata": metadata or {},
        }
        joblib.dump(bundle, path)
        return path


def load_holt_winters_model(path: str):
    import joblib

    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        forecaster = HoltWintersForecaster(
            trend=bundle.get("trend", "add"),
            seasonal=bundle.get("seasonal", "add"),
            seasonal_periods=bundle.get("seasonal_periods", 7),
            damped_trend=bundle.get("damped_trend", False),
            use_log=bundle.get("use_log", True),
        )
        forecaster.model_ = bundle["model"]
        forecaster.series_ = pd.Series([], name=bundle.get("series_name", "demand"))
        return forecaster
    raise TypeError(f"Unsupported model bundle format in {path}")

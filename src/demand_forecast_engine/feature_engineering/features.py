"""Calendar, lag and rolling features, plus a strict temporal split.

Leakage rules followed here:
  * lag and rolling features are computed inside each series (never across series)
  * rolling windows are shifted by one step, so a row never sees its own value
  * the split is by date: train is everything up to the cutoff, test is everything after
  * category codes are learned from the train set only
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from demand_forecast_engine.utils import DataValidationError


def assert_sorted(df: pd.DataFrame, group_cols: Sequence[str], date_col: str = "Date") -> None:
    """Lag/rolling features need rows ordered by series, then by date."""
    keys = df[[*group_cols, date_col]].reset_index(drop=True)
    expected = keys.sort_values(list(keys.columns), kind="stable").reset_index(drop=True)
    if not keys.equals(expected):
        raise DataValidationError("Data must be sorted by group columns and date (run preprocess.py first)")


def add_calendar_features(df: pd.DataFrame, date_col: str, features: Sequence[str]) -> pd.DataFrame:
    """Add calendar columns derived from the date. These are known in advance, so safe."""
    dt = df[date_col].dt
    available = {
        "day": dt.day,
        "month": dt.month,
        "week": dt.isocalendar().week.astype(int),
        "dayofweek": dt.dayofweek,
        "year": dt.year,
        "quarter": dt.quarter,
        "is_weekend": (dt.dayofweek >= 5).astype(int),
    }
    unknown = set(features) - set(available)
    if unknown:
        raise DataValidationError(f"Unknown calendar features: {sorted(unknown)}. Options: {sorted(available)}")
    df = df.copy()
    for name in features:
        df[name] = available[name]
    return df


def add_lag_features(
    df: pd.DataFrame, cols: Sequence[str], lags: Sequence[int], group_cols: Sequence[str]
) -> pd.DataFrame:
    """Add `<col>_lag_<n>`: the value n rows earlier in the same series."""
    df = df.copy()
    grouped = df.groupby(list(group_cols), sort=False)
    for col in cols:
        for lag in lags:
            if lag < 1:
                raise DataValidationError("Lags must be >= 1 (lag 0 would be the target itself)")
            df[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int],
    stats: Sequence[str],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Add `<col>_roll_<stat>_<window>`, computed on past values only (shift by 1)."""
    allowed = {"mean", "std", "min", "max"}
    bad = set(stats) - allowed
    if bad:
        raise DataValidationError(f"Unsupported rolling stats: {sorted(bad)}. Options: {sorted(allowed)}")

    df = df.copy()
    for col in cols:
        grouped_col = df.groupby(list(group_cols), sort=False)[col]
        for window in windows:
            for stat in stats:
                df[f"{col}_roll_{stat}_{window}"] = grouped_col.transform(
                    lambda s, w=window, st=stat: getattr(s.shift(1).rolling(w), st)()
                )
    return df


def resolve_cutoff(df: pd.DataFrame, date_col: str, cutoff_date=None, test_ratio=None) -> pd.Timestamp:
    """Last train date. An explicit cutoff date wins, otherwise the last `test_ratio` of the
    calendar days (by time, never random) becomes the test set."""
    if cutoff_date is not None:
        return pd.Timestamp(cutoff_date)
    if test_ratio is None:
        raise DataValidationError("Set features.test_ratio or features.train_cutoff_date in params.yaml")
    if not 0 < test_ratio < 1:
        raise DataValidationError(f"features.test_ratio must be between 0 and 1, got {test_ratio}")
    days = df[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n_test = max(1, math.ceil(round(test_ratio * len(days), 9)))
    if n_test >= len(days):
        raise DataValidationError(f"test_ratio {test_ratio} leaves no training days ({len(days)} days in total)")
    return pd.Timestamp(days.iloc[-n_test - 1])


def temporal_split(df: pd.DataFrame, date_col: str, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = Date <= cutoff, test = Date > cutoff. Never shuffles."""
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df[date_col] <= cutoff_ts].copy()
    test = df[df[date_col] > cutoff_ts].copy()
    if train.empty or test.empty:
        raise DataValidationError(f"Cutoff {cutoff} leaves an empty train or test set")
    if train[date_col].max() >= test[date_col].min():
        raise AssertionError("Train and test dates overlap")
    return train, test


def fit_category_maps(train: pd.DataFrame, cols: Sequence[str]) -> dict[str, dict[str, int]]:
    """Learn a category -> integer code map from the train set only."""
    return {
        col: {str(v): i for i, v in enumerate(sorted(train[col].dropna().astype(str).unique()))}
        for col in cols
    }


def apply_category_maps(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Encode columns with the learned maps. Categories unseen in train become -1."""
    df = df.copy()
    for col, mapping in maps.items():
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df

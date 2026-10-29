"""Cleaning steps moved out of the Preprocessing notebook.

Every step here only looks at past values of a series, or at one row at a time,
so running them before the train/test split does not leak test information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from demand_forecast_engine.utils import DataValidationError

log = logging.getLogger(__name__)


def load_raw(path) -> pd.DataFrame:
    """Read the raw CSV as it is. Dates are parsed later, after the schema check."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_dates(df: pd.DataFrame, date_col: str, date_format: str) -> pd.DataFrame:
    """Parse the date column, failing with examples of the values that do not match."""
    parsed = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        examples = list(df.loc[bad, date_col].astype(str).unique()[:3])
        raise DataValidationError(
            f"Column '{date_col}': {int(bad.sum())} value(s) are empty or do not match the "
            f"format '{date_format}', e.g. {examples}"
        )
    df = df.copy()
    df[date_col] = parsed
    return df


def sort_temporal(
    df: pd.DataFrame, group_cols: Sequence[str], date_col: str = "Date"
) -> pd.DataFrame:
    """Sort by series, then by date, so each series is in time order."""
    return df.sort_values([*group_cols, date_col], kind="stable").reset_index(drop=True)


def clean_data(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    target_col: str,
    ffill_cols: Sequence[str],
) -> pd.DataFrame:
    """Sort, drop duplicate rows, forward-fill gaps per series, drop rows with no target."""
    group_cols, ffill_cols = list(group_cols), list(ffill_cols)

    df = sort_temporal(df, group_cols, date_col)

    n_before = len(df)
    df = df.drop_duplicates(subset=[*group_cols, date_col], keep="first")
    if len(df) < n_before:
        log.warning("Dropped %d duplicate (series, date) rows", n_before - len(df))

    # Forward fill uses only earlier values in the same series.
    if ffill_cols:
        df[ffill_cols] = df.groupby(group_cols)[ffill_cols].ffill()

    n_before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < n_before:
        log.info("Dropped %d rows where %s is still missing", n_before - len(df), target_col)

    return df.reset_index(drop=True)


def add_log1p(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Add a `<col>_log1p` column for each column. Invert later with np.expm1."""
    df = df.copy()
    for col in cols:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' has negative values, log1p is not valid")
        df[f"{col}_log1p"] = np.log1p(df[col])
    return df

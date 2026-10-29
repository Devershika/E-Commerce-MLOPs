"""Data quality checks. Every check raises DataValidationError with a message that says
what is wrong and where, so a failing pipeline tells you what to fix."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from demand_forecast_engine.utils import DataValidationError

log = logging.getLogger(__name__)


def _preview(items, limit: int = 5) -> str:
    items = list(items)
    shown = ", ".join(str(i) for i in items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def require_columns(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    """Fail with the list of missing columns, plus the columns that were found."""
    missing = [c for c in dict.fromkeys(cols) if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{context}: missing required column(s): {_preview(missing, 10)}. "
            f"Columns found: {_preview(df.columns, 20)}"
        )


def validate_schema(df: pd.DataFrame, schema: Mapping[str, Sequence[str]]) -> None:
    """Check that the schema columns exist and have the expected kind of values."""
    date_cols = list(schema.get("date", []))
    string_cols = list(schema.get("string", []))
    numeric_cols = list(schema.get("numeric", []))
    require_columns(df, [*date_cols, *string_cols, *numeric_cols], "Raw data")

    problems = []
    for col in numeric_cols:
        if not is_numeric_dtype(df[col]):
            not_numbers = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            examples = _preview(df.loc[not_numbers, col].astype(str).unique(), 3)
            problems.append(
                f"'{col}' must be numeric but has dtype {df[col].dtype} (e.g. {examples})"
            )
    for col in string_cols:
        if is_numeric_dtype(df[col]):
            problems.append(f"'{col}' must be text but has numeric dtype {df[col].dtype}")
    if problems:
        raise DataValidationError(
            "Raw data has wrong column types:\n  - " + "\n  - ".join(problems)
        )


def validate_nulls(
    df: pd.DataFrame,
    cols: Sequence[str],
    context: str,
    max_fraction: float = 0.0,
    overrides: Mapping[str, float] | None = None,
) -> None:
    """Fail if a column has more missing values than allowed (default: none)."""
    overrides = overrides or {}
    problems = []
    for col in cols:
        allowed = overrides.get(col, max_fraction)
        n_null = int(df[col].isna().sum())
        share = n_null / len(df) if len(df) else 0.0
        if share > allowed:
            problems.append(f"'{col}': {n_null} missing ({share:.2%}), allowed {allowed:.2%}")
    if problems:
        raise DataValidationError(
            f"{context}: too many missing values:\n  - " + "\n  - ".join(problems)
        )


def validate_non_negative(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    """Fail if a column has negative values (log1p is undefined below -1)."""
    problems = [
        f"'{c}': {int((df[c] < 0).sum())} negative value(s)" for c in cols if (df[c] < 0).any()
    ]
    if problems:
        raise DataValidationError(
            f"{context}: negative values found:\n  - " + "\n  - ".join(problems)
        )


def validate_date_continuity(
    df: pd.DataFrame, group_cols: Sequence[str], date_col: str, strict: bool = True
) -> None:
    """Every series must have one row per day between its first and last date."""
    span = df.groupby(list(group_cols))[date_col].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    broken = span[span["count"] != expected].copy()
    if broken.empty:
        return
    broken["expected"] = expected[broken.index]
    lines = [
        f"{key}: expected {int(row['expected'])} daily rows, found {int(row['count'])}"
        for key, row in broken.head(5).iterrows()
    ]
    message = (
        f"Timestamp continuity: {len(broken)} of {len(span)} series have missing or repeated days "
        f"(lags assume one row per day):\n  - " + "\n  - ".join(lines)
    )
    if strict:
        raise DataValidationError(
            message + "\nFix the data, or set preprocess.require_continuous_dates: false."
        )
    log.warning(message)


def validate_model_input(
    train: pd.DataFrame, test: pd.DataFrame, group_cols: Sequence[str], min_rows_per_series: int
) -> None:
    """Final checks on the files that models will read."""
    if list(train.columns) != list(test.columns):
        raise DataValidationError("Train and test have different columns")

    for name, part in (("train", train), ("test", test)):
        numbers = part.select_dtypes("number")
        bad = {c: int(n) for c, n in numbers.isna().sum().items() if n}
        bad.update({c: int(n) for c, n in np.isinf(numbers).sum().items() if n})
        if bad:
            raise DataValidationError(f"{name}: missing or infinite values in {bad}")

    sizes = train.groupby(list(group_cols)).size()
    short = sizes[sizes < min_rows_per_series]
    if not short.empty:
        raise DataValidationError(
            f"{len(short)} series have fewer than {min_rows_per_series} train rows "
            f"(2 x model.seasonal_periods), e.g. {_preview(short.index, 3)}"
        )

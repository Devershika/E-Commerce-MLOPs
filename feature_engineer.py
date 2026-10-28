#!/usr/bin/env python
"""Step 2: build features and split into train/test by time.

Reads the cleaned parquet from preprocess.py, adds calendar, lag and rolling
features, splits by time and writes train.parquet and test.parquet. On bad input
it prints a clear error and exits with code 1.

The test set is the last `features.test_ratio` of the calendar days, unless a
`features.train_cutoff_date` is set.

Usage:
    python feature_engineer.py
    python feature_engineer.py --test-ratio 0.1
    python feature_engineer.py --cutoff-date 2023-10-31
    python feature_engineer.py --params params.yaml --verbose
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast_engine.feature_engineering.features import (  # noqa: E402
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    apply_category_maps,
    assert_sorted,
    fit_category_maps,
    resolve_cutoff,
    temporal_split,
)
from demand_forecast_engine.preprocessing.validate import (  # noqa: E402
    require_columns,
    validate_model_input,
)
from demand_forecast_engine.utils import load_params, run_cli  # noqa: E402

log = logging.getLogger("feature_engineer")


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default="params.yaml", help="Parameter file (default: params.yaml)")
    p.add_argument("--input", help="Clean parquet path (overrides paths.clean_data)")
    p.add_argument("--train-output", help="Train parquet path (overrides paths.train_data)")
    p.add_argument("--test-output", help="Test parquet path (overrides paths.test_data)")
    p.add_argument("--cutoff-date", help="Last train date, YYYY-MM-DD (overrides test_ratio)")
    p.add_argument("--test-ratio", type=float, help="Share of calendar days used as test (overrides params.yaml)")
    p.add_argument("--verbose", action="store_true", help="Show the full traceback on errors")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    params = load_params(resolve(args.params))
    data, feat, model, paths = params["data"], params["features"], params["model"], params["paths"]
    date_col, group_cols = data["date_col"], data["group_cols"]

    src = resolve(args.input or paths["clean_data"])
    train_path = resolve(args.train_output or paths["train_data"])
    test_path = resolve(args.test_output or paths["test_data"])
    maps_path = resolve(paths["category_maps"])

    if args.cutoff_date:
        cutoff_date, test_ratio = args.cutoff_date, None
    elif args.test_ratio is not None:
        cutoff_date, test_ratio = None, args.test_ratio
    else:
        cutoff_date, test_ratio = feat.get("train_cutoff_date"), feat.get("test_ratio")

    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    log.info("Loaded %s: %d rows", src, len(df))

    needed = [
        date_col, *group_cols, feat["target_col"], *feat["lag_cols"], *feat["rolling_cols"],
        *feat["categorical_features"], *feat["drop_cols"],
    ]
    require_columns(df, needed, f"Clean data ({src.name}), run preprocess.py again?")
    assert_sorted(df, group_cols, date_col)

    # The cutoff is chosen from the full timeline, before warm-up rows are dropped.
    cutoff = resolve_cutoff(df, date_col, cutoff_date, test_ratio)

    # 1. Features that only look at the past or at the calendar.
    df = add_calendar_features(df, date_col, feat["calendar"])
    before = set(df.columns)
    df = add_lag_features(df, feat["lag_cols"], feat["lag_days"], group_cols)
    df = add_rolling_features(df, feat["rolling_cols"], feat["rolling_windows"], feat["rolling_stats"], group_cols)
    history_cols = sorted(set(df.columns) - before)

    # 2. Drop the first rows of each series, which have no history yet.
    n_before = len(df)
    df = df.dropna(subset=history_cols).reset_index(drop=True)
    log.info("Dropped %d warm-up rows without full history", n_before - len(df))

    # 3. Drop same-day columns that would leak the target (after they were lagged).
    df = df.drop(columns=feat["drop_cols"])

    # 4. Time-based split, then learn category codes on train only.
    train, test = temporal_split(df, date_col, cutoff)
    maps = fit_category_maps(train, feat["categorical_features"])
    train, test = apply_category_maps(train, maps), apply_category_maps(test, maps)

    # 5. Final checks before anything is written.
    validate_model_input(train, test, group_cols, min_rows_per_series=2 * model["seasonal_periods"])

    for path in (train_path, test_path, maps_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    maps_path.write_text(json.dumps(maps, indent=2), encoding="utf-8")

    log.info("Cutoff (last train date): %s", cutoff.date())
    log.info("Train: %d rows, %s to %s", len(train), train[date_col].min().date(), train[date_col].max().date())
    log.info("Test:  %d rows, %s to %s", len(test), test[date_col].min().date(), test[date_col].max().date())
    log.info("Wrote %s, %s and %s", train_path, test_path, maps_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_cli(lambda: run(args), log, args.verbose)


if __name__ == "__main__":
    main()

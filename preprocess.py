#!/usr/bin/env python
"""Step 1: validate and clean the raw sales CSV.

Checks the schema, null values and date continuity, sorts each series by date,
forward-fills gaps in Demand, adds a log1p version of Demand and writes a cleaned
parquet file. On bad data it prints a clear error and exits with code 1.

Usage:
    python preprocess.py
    python preprocess.py --input Data/sales_data.csv --output Data/interim/clean.parquet
    python preprocess.py --params params.yaml --verbose
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast_engine.preprocessing.clean import (  # noqa: E402
    add_log1p,
    clean_data,
    load_raw,
    parse_dates,
)
from demand_forecast_engine.preprocessing.validate import (  # noqa: E402
    validate_date_continuity,
    validate_non_negative,
    validate_nulls,
    validate_schema,
)
from demand_forecast_engine.utils import load_params, run_cli  # noqa: E402

log = logging.getLogger("preprocess")


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default="params.yaml", help="Parameter file (default: params.yaml)")
    p.add_argument("--input", help="Raw CSV path (overrides paths.raw_data)")
    p.add_argument("--output", help="Clean parquet path (overrides paths.clean_data)")
    p.add_argument("--verbose", action="store_true", help="Show the full traceback on errors")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    params = load_params(resolve(args.params))
    data, schema, prep, paths = params["data"], params["schema"], params["preprocess"], params["paths"]
    date_col, group_cols = data["date_col"], data["group_cols"]
    src = resolve(args.input or paths["raw_data"])
    dst = resolve(args.output or paths["clean_data"])

    df = load_raw(src)
    log.info("Loaded %s: %d rows, %d columns", src, *df.shape)

    # Checks on the raw file (before anything is filled in)
    validate_schema(df, schema)
    required = [*schema.get("date", []), *schema.get("string", []), *schema.get("numeric", [])]
    validate_nulls(
        df, required, "Raw data",
        overrides={c: prep["max_null_fraction"] for c in prep["ffill_cols"]},
    )
    df = parse_dates(df, date_col, data["date_format"])

    df = clean_data(df, group_cols, date_col, data["target_col"], prep["ffill_cols"])

    # Checks on the cleaned data
    validate_nulls(df, required, "After cleaning")
    validate_non_negative(df, prep["log1p_cols"], "Before log1p")
    validate_date_continuity(df, group_cols, date_col, strict=prep["require_continuous_dates"])

    df = add_log1p(df, prep["log1p_cols"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    log.info("Wrote %s: %d rows", dst, len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_cli(lambda: run(args), log, args.verbose)


if __name__ == "__main__":
    main()

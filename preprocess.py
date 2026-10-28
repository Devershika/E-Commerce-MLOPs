#!/usr/bin/env python
"""Step 1: clean the raw sales CSV.

Reads the raw data, sorts it by series and date, forward-fills gaps in Demand,
adds a log1p version of Demand and writes a cleaned parquet file.

Usage:
    python preprocess.py
    python preprocess.py --input Data/sales_data.csv --output Data/interim/clean.parquet
    python preprocess.py --params params.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast_engine.preprocessing.clean import (  # noqa: E402
    add_log1p,
    check_regular_frequency,
    clean_data,
    load_raw,
)
from demand_forecast_engine.utils import load_params  # noqa: E402

log = logging.getLogger("preprocess")


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default="params.yaml", help="Parameter file (default: params.yaml)")
    p.add_argument("--input", help="Raw CSV path (overrides paths.raw_data)")
    p.add_argument("--output", help="Clean parquet path (overrides paths.clean_data)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    params = load_params(resolve(args.params))

    data, prep, paths = params["data"], params["preprocess"], params["paths"]
    src = resolve(args.input or paths["raw_data"])
    dst = resolve(args.output or paths["clean_data"])

    df = load_raw(src, data["date_col"], data["date_format"])
    log.info("Loaded %s: %d rows, %d columns", src, *df.shape)

    df = clean_data(
        df,
        group_cols=data["group_cols"],
        date_col=data["date_col"],
        target_col=data["target_col"],
        ffill_cols=prep["ffill_cols"],
    )
    df = add_log1p(df, prep["log1p_cols"])
    check_regular_frequency(df, data["group_cols"], data["date_col"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    log.info("Wrote %s: %d rows", dst, len(df))


if __name__ == "__main__":
    main()

"""Quality checks: failures must be clear, and must stop the scripts with exit code 1."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast_engine.feature_engineering.features import resolve_cutoff  # noqa: E402
from demand_forecast_engine.preprocessing.clean import parse_dates  # noqa: E402
from demand_forecast_engine.preprocessing.validate import (  # noqa: E402
    require_columns,
    validate_date_continuity,
    validate_model_input,
    validate_non_negative,
    validate_nulls,
    validate_schema,
)
from demand_forecast_engine.utils import DataValidationError  # noqa: E402

SCHEMA = {"date": ["Date"], "string": ["Store ID"], "numeric": ["Price", "Demand"]}


def raw(n=10):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=n).strftime("%d-%m-%Y"),
            "Store ID": "S1",
            "Price": np.arange(n, dtype=float),
            "Demand": np.arange(n, dtype=float),
        }
    )


def test_missing_column_is_named_in_the_error():
    df = raw().drop(columns=["Price"])
    with pytest.raises(DataValidationError, match=r"missing required column\(s\): Price"):
        validate_schema(df, SCHEMA)


def test_wrong_type_is_reported_with_an_example():
    df = raw()
    df["Price"] = ["abc"] + list(range(9))
    with pytest.raises(DataValidationError, match=r"'Price' must be numeric.*abc"):
        validate_schema(df, SCHEMA)


def test_bad_date_format_shows_the_value_and_expected_format():
    df = raw()
    df.loc[3, "Date"] = "2023/13/45"
    with pytest.raises(DataValidationError, match=r"2023/13/45"):
        parse_dates(df, "Date", "%d-%m-%Y")


def test_null_tolerance_per_column():
    df = raw()
    df.loc[0, "Demand"] = np.nan  # 10% missing
    with pytest.raises(DataValidationError, match="'Demand'"):
        validate_nulls(df, ["Demand"], "Raw data", max_fraction=0.05)
    validate_nulls(df, ["Demand"], "Raw data", overrides={"Demand": 0.2})  # allowed, no error


def test_negative_values_are_rejected():
    df = raw()
    df.loc[2, "Demand"] = -1
    with pytest.raises(DataValidationError, match="negative"):
        validate_non_negative(df, ["Demand"], "Before log1p")


def test_date_gap_is_reported_per_series_and_can_be_downgraded_to_a_warning(caplog):
    df = raw(10).drop(index=4)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    with pytest.raises(DataValidationError, match=r"expected 10 daily rows, found 9"):
        validate_date_continuity(df, ["Store ID"], "Date", strict=True)
    validate_date_continuity(df, ["Store ID"], "Date", strict=False)  # warns only
    assert "missing or repeated days" in caplog.text


def test_test_ratio_picks_the_last_days_by_time():
    df = pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=100)})
    cutoff = resolve_cutoff(df, "Date", test_ratio=0.1)
    assert cutoff == pd.Timestamp("2023-03-31")  # 10 test days: Apr 1 to Apr 10
    assert resolve_cutoff(df, "Date", cutoff_date="2023-02-01", test_ratio=0.1) == pd.Timestamp("2023-02-01")


@pytest.mark.parametrize("ratio", [0, 1, 1.5, -0.1])
def test_bad_test_ratio_is_rejected(ratio):
    df = pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=100)})
    with pytest.raises(DataValidationError, match="between 0 and 1"):
        resolve_cutoff(df, "Date", test_ratio=ratio)


def test_series_with_too_little_history_are_reported():
    train = pd.DataFrame({"Store ID": ["S1"] * 20 + ["S2"] * 5, "x": 1.0})
    test = train.copy()
    with pytest.raises(DataValidationError, match="fewer than 14 train rows"):
        validate_model_input(train, test, ["Store ID"], min_rows_per_series=14)


def test_require_columns_lists_what_was_found():
    with pytest.raises(DataValidationError, match="Columns found: Date, Store ID"):
        require_columns(raw()[["Date", "Store ID"]], ["Date", "Demand"], "Clean data")


# ---- the scripts themselves: one clear line, exit code 1, no traceback --------------------------

def run_script(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True)


def test_preprocess_script_fails_gracefully_on_missing_column(tmp_path):
    import yaml

    schema = yaml.safe_load((ROOT / "params.yaml").read_text())["schema"]
    columns = [c for kind in ("date", "string", "numeric") for c in schema[kind]]
    bad = tmp_path / "bad.csv"
    pd.DataFrame([["01-01-2023"] + ["x"] * 6 + [1] * 9], columns=columns).drop(columns=["Price"]).to_csv(bad, index=False)
    result = run_script("preprocess.py", "--input", str(bad), "--output", str(tmp_path / "out.parquet"))
    assert result.returncode == 1
    assert "missing required column(s): Price" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "out.parquet").exists()  # nothing is written when validation fails


def test_preprocess_script_fails_gracefully_on_missing_file(tmp_path):
    result = run_script("preprocess.py", "--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "o.parquet"))
    assert result.returncode == 1
    assert "dvc pull" in result.stderr and "Traceback" not in result.stderr


def test_feature_script_fails_gracefully_when_input_lacks_columns(tmp_path):
    bad = tmp_path / "clean.parquet"
    pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=5)}).to_parquet(bad)
    result = run_script(
        "feature_engineer.py", "--input", str(bad),
        "--train-output", str(tmp_path / "tr.parquet"), "--test-output", str(tmp_path / "te.parquet"),
    )
    assert result.returncode == 1
    assert "missing required column(s)" in result.stderr and "Traceback" not in result.stderr

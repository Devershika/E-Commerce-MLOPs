"""Leakage checks for the preprocessing and feature engineering steps."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast_engine.feature_engineering.features import (  # noqa: E402
    add_lag_features,
    add_rolling_features,
    apply_category_maps,
    fit_category_maps,
    temporal_split,
)
from demand_forecast_engine.preprocessing.clean import add_log1p, clean_data  # noqa: E402

GROUPS = ["Store ID"]


def make_df(n=40):
    frames = []
    for store, base in [("S1", 10), ("S2", 1000)]:
        frames.append(
            pd.DataFrame(
                {
                    "Store ID": store,
                    "Date": pd.date_range("2023-01-01", periods=n),
                    "Demand": base + np.arange(n, dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_lag_stays_inside_each_series():
    df = add_lag_features(make_df(), ["Demand"], [1], GROUPS)
    s1, s2 = df[df["Store ID"] == "S1"], df[df["Store ID"] == "S2"]
    assert np.isnan(s2["Demand_lag_1"].iloc[0])  # not filled with S1's last value
    assert (s1["Demand_lag_1"].iloc[1:].values == s1["Demand"].iloc[:-1].values).all()


def test_rolling_uses_only_past_values_and_never_crosses_series():
    df = add_rolling_features(make_df(), ["Demand"], [3], ["mean"], GROUPS)
    s1 = df[df["Store ID"] == "S1"].reset_index(drop=True)
    s2 = df[df["Store ID"] == "S2"].reset_index(drop=True)
    assert s1["Demand_roll_mean_3"].iloc[:3].isna().all()
    assert s1["Demand_roll_mean_3"].iloc[5] == s1["Demand"].iloc[2:5].mean()  # excludes row 5
    assert s2["Demand_roll_mean_3"].iloc[:3].isna().all()  # no S1 values leak in


def test_changing_a_value_does_not_change_features_of_earlier_or_same_row():
    base = make_df()
    changed = base.copy()
    changed.loc[changed.index[39], "Demand"] = 1e6  # last row of S1

    def feats(d):
        d = add_lag_features(d, ["Demand"], [1, 7], GROUPS)
        return add_rolling_features(d, ["Demand"], [7], ["mean", "std"], GROUPS)

    a, b = feats(base), feats(changed)
    feature_cols = [c for c in a.columns if "lag" in c or "roll" in c]
    pd.testing.assert_frame_equal(a.loc[:39, feature_cols], b.loc[:39, feature_cols])


def test_split_is_strictly_by_time():
    train, test = temporal_split(make_df(), "Date", "2023-01-30")
    assert train["Date"].max() <= pd.Timestamp("2023-01-30") < test["Date"].min()
    assert len(train) + len(test) == len(make_df())


def test_split_rejects_empty_sets():
    with pytest.raises(ValueError):
        temporal_split(make_df(), "Date", "2030-01-01")


def test_category_codes_come_from_train_only():
    train = pd.DataFrame({"Weather": ["Rainy", "Sunny"]})
    test = pd.DataFrame({"Weather": ["Sunny", "Hail"]})
    maps = fit_category_maps(train, ["Weather"])
    assert apply_category_maps(test, maps)["Weather"].tolist() == [1, -1]


def test_clean_forward_fills_per_series_and_log1p_roundtrips():
    df = make_df()
    df.loc[[5, 45], "Demand"] = np.nan
    df.loc[40, "Demand"] = np.nan  # first row of S2: nothing earlier in its series to fill from
    out = clean_data(df, GROUPS, "Date", "Demand", ["Demand"])
    assert len(out) == len(df) - 1  # only the unfillable first row of S2 is dropped
    assert out["Demand"].notna().all()
    out = add_log1p(out, ["Demand"])
    np.testing.assert_allclose(np.expm1(out["Demand_log1p"]), out["Demand"])

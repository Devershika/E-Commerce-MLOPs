from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demand_forecast_engine.models.holt_winters import HoltWintersForecaster


def load_sales_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the input dataset.")
    if "Demand" not in df.columns:
        raise ValueError("Expected a 'Demand' column in the input dataset.")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a Holt-Winters demand forecast model.")
    parser.add_argument("--input", default="Data/sales_data.csv", help="CSV input file containing date and demand columns.")
    parser.add_argument("--output", default="models/holt_winters_model.joblib", help="Destination for the serialized model bundle.")
    parser.add_argument("--group-by", nargs="*", default=["Category", "Region", "Store ID", "Product ID"], help="Columns used to group the series.")
    parser.add_argument("--seasonal-periods", type=int, default=7, help="Seasonal period used by Holt-Winters.")
    args = parser.parse_args()

    df = load_sales_data(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_cols = [col for col in args.group_by if col in df.columns]
    if not group_cols:
        raise ValueError("No valid grouping columns were found in the input data.")

    group_records = []
    for key, group in df.groupby(group_cols, dropna=False):
        ts = group.sort_values("Date").set_index("Date")["Demand"].astype(float)
        if len(ts) < 30:
            continue

        train_end = int(len(ts) * 0.8)
        train = ts.iloc[:train_end]
        model = HoltWintersForecaster(
            trend="add",
            seasonal="add",
            seasonal_periods=args.seasonal_periods,
            use_log=True,
        ).fit(train)
        forecast = model.forecast(len(ts) - train_end)
        group_records.append({"group": key, "model": model, "forecast": forecast})

    if not group_records:
        raise ValueError("No valid series were found to train the model.")

    best_group = max(group_records, key=lambda item: len(item["forecast"]))
    bundle = {
        "model": best_group["model"].model_,
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": args.seasonal_periods,
        "use_log": True,
        "metadata": {"group_columns": group_cols, "trained_group": best_group["group"]},
    }

    import joblib

    joblib.dump(bundle, output_path)
    print(f"Model bundle saved to {output_path}")


if __name__ == "__main__":
    main()

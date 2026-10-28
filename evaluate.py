from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from demand_forecast_engine.models.holt_winters import HoltWintersForecaster


def load_sales_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the input dataset.")
    if "Demand" not in df.columns:
        raise ValueError("Expected a 'Demand' column in the input dataset.")

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


def evaluate_model(csv_path: str, output_path: str) -> dict:
    df = load_sales_data(csv_path)
    group_cols = [col for col in ["Category", "Region", "Store ID", "Product ID"] if col in df.columns]
    if not group_cols:
        raise ValueError("Dataset does not include the expected product grouping columns.")

    results = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        ts = group.sort_values("Date").set_index("Date")["Demand"].astype(float)
        if len(ts) < 30:
            continue

        split_index = int(len(ts) * 0.8)
        train = ts.iloc[:split_index]
        test = ts.iloc[split_index:]

        model = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=7, use_log=True)
        model.fit(train)
        forecast = model.forecast(len(test))
        actual = test.reindex(forecast.index, fill_value=np.nan).dropna()
        forecast = forecast.reindex(actual.index)

        if len(actual) == 0:
            continue

        mape = mean_absolute_percentage_error(actual.to_numpy(), forecast.to_numpy())
        r2 = r2_score(actual.to_numpy(), forecast.to_numpy())
        results.append({
            "group": dict(zip(group_cols, group_key)),
            "MAPE": float(mape),
            "R2": float(r2),
            "sample_count": int(len(actual)),
        })

    output = {
        "model": "Holt-Winters (Fine-Tuned)",
        "series_evaluated": len(results),
        "summary": {
            "mean_mape": float(np.mean([r["MAPE"] for r in results])) if results else 0.0,
            "mean_r2": float(np.mean([r["R2"] for r in results])) if results else 0.0,
            "max_mape": float(np.max([r["MAPE"] for r in results])) if results else 0.0,
        },
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Holt-Winters demand model and save metrics.")
    parser.add_argument("--input", default="Data/sales_data.csv", help="CSV dataset for evaluation.")
    parser.add_argument("--output", default="reports/metrics.json", help="Output filename for the JSON metrics report.")
    args = parser.parse_args()

    metrics = evaluate_model(args.input, args.output)
    print(json.dumps({"series_evaluated": metrics["series_evaluated"], "mean_mape": metrics["summary"]["mean_mape"]}, indent=2))


if __name__ == "__main__":
    main()

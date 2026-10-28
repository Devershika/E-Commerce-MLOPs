from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_group_series(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        return df, df.iloc[0:0]

    split_index = max(1, int(len(df) * (1 - test_ratio)))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def build_processed_data(input_path: str, output_dir: str, test_ratio: float) -> tuple[Path, Path]:
    df = pd.read_csv(input_path)
    if "Date" not in df.columns or "Demand" not in df.columns:
        raise ValueError("Raw dataset must contain 'Date' and 'Demand' columns.")

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for _, group in df.groupby(["Store ID", "Product ID", "Category", "Region"], dropna=False):
        if len(group) < 30:
            continue
        group_train, group_test = split_group_series(group, test_ratio=test_ratio)
        train_frames.append(group_train)
        test_frames.append(group_test)

    if not train_frames or not test_frames:
        raise ValueError("No valid time series groups were found for a train/test split.")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    train_path = out_dir_path / "train.parquet"
    test_path = out_dir_path / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed train/test parquet files for model training.")
    parser.add_argument("--input", default="Data/sales_data.csv", help="Input CSV file path.")
    parser.add_argument("--output", default="data/processed", help="Directory for processed parquet outputs.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Train/test split ratio per Store×Product series.")
    args = parser.parse_args()

    train_path, test_path = build_processed_data(args.input, args.output, args.test_ratio)
    print(f"Train split saved to {train_path}")
    print(f"Test split saved to {test_path}")


if __name__ == "__main__":
    main()

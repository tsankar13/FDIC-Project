from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_and_merge_csvs(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path).dropna(axis=1, how="all")
        if "CERT" not in df.columns:
            raise ValueError(f"Missing CERT column in {path.name}")
        frames.append(df.set_index("CERT"))

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from merged dataset: {missing}")

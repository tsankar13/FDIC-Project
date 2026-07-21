"""Output helpers: timestamped run directories and JSON/CSV artifact writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def create_run_dir(output_dir: Path, mode: str) -> Path:
    """Create and return a fresh ``<UTC-timestamp>_<mode>`` directory for a run."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{stamp}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(payload: dict, path: Path) -> None:
    """Write ``payload`` as sorted, indented JSON for stable, diff-friendly output."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_frame(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, preserving the index (the ``CERT`` key)."""
    df.to_csv(path, index=True)

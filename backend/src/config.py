"""Run configuration: the ``PipelineConfig`` dataclass and JSON loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Typed settings for a pipeline run (paths, mode, seed, and tuning knobs)."""

    input_dir: Path
    output_dir: Path = Path("backend/artifacts")
    mode: str = "clustering"
    random_seed: int = 42
    max_clusters: int = 8
    outlier_quantile_low: float = 0.01
    outlier_quantile_high: float = 0.99
    label_column: str = "FAILED"
    construct_label_if_missing: bool = True
    feature_columns_required: list[str] = field(
        default_factory=lambda: ["RBCT1J", "ASSET", "INTINC", "EINTEXP", "NAASSET", "P9ASSET", "NTLS"]
    )

    @classmethod
    def from_dict(cls, values: dict) -> "PipelineConfig":
        """Build a config from a JSON-style dict, coercing path fields to ``Path``."""
        payload = dict(values)
        payload["input_dir"] = Path(payload["input_dir"])
        payload["output_dir"] = Path(payload.get("output_dir", "backend/artifacts"))
        return cls(**payload)

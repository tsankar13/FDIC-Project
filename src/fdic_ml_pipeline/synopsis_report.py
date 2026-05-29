from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _latest_synopsis_run(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_*/synopsis_metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No synopsis artifacts found under {output_dir}")
    return candidates[-1].parent


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pick_primary_structure(metrics: dict) -> tuple[str, float]:
    kmeans = float(metrics["kmeans_k2_silhouette"])
    agg = float(metrics["agglomerative_k2_silhouette"])
    if kmeans >= agg:
        return "KMeans (k=2)", kmeans
    return "Agglomerative clustering (k=2)", agg


def generate_synopsis_report(
    output_dir: Path = Path("backend/artifacts"),
    reports_dir: Path = Path("backend/artifacts/reports/primary"),
    run_dir: Path | None = None,
) -> Path:
    selected_run_dir = run_dir if run_dir is not None else _latest_synopsis_run(output_dir)
    metrics_path = selected_run_dir / "synopsis_metrics.json"
    correlations_path = selected_run_dir / "synopsis_correlations.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing synopsis metrics file: {metrics_path}")
    if not correlations_path.exists():
        raise FileNotFoundError(f"Missing synopsis correlations file: {correlations_path}")

    metrics = json.loads(metrics_path.read_text())
    correlations = pd.read_csv(correlations_path)

    top_pairs = correlations.head(3).to_dict(orient="records")
    structure_method, structure_score = _pick_primary_structure(metrics)
    now_stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"DATA_SYNOPSIS_AUTOREPORT_{now_stamp}.md"

    pair_lines = []
    for pair in top_pairs:
        pair_lines.append(
            "- "
            f"`{pair['feature_left']}` vs `{pair['feature_right']}`: "
            f"correlation `{float(pair['correlation']):.4f}`"
        )
    if not pair_lines:
        pair_lines.append("- No correlation pairs were available in the artifact.")

    text = "\n".join(
        [
            "# Automated Data Synopsis Report",
            "",
            "## Overview",
            f"This auto-generated synopsis summarizes the latest ML-based structural diagnostics from `{selected_run_dir}`.",
            f"The analysis covers `{int(metrics['rows_analyzed'])}` institutions and `{int(metrics['features_analyzed'])}` engineered risk features.",
            "",
            "## Narrative Synopsis",
            (
                "The feature space is compressible and structured: "
                f"the top principal component explains `{_format_pct(float(metrics['pca_explained_variance_pc1']))}` "
                f"of variance, while the first `{int(metrics['pca_components'])}` components explain "
                f"`{_format_pct(float(metrics['pca_explained_variance_top3']))}`. "
                "This indicates that risk variation is driven by a relatively compact set of latent factors."
            ),
            (
                f"Cluster-separation diagnostics show the strongest two-group structure under `{structure_method}` "
                f"with silhouette `{structure_score:.4f}` (KMeans: `{float(metrics['kmeans_k2_silhouette']):.4f}`, "
                f"Agglomerative: `{float(metrics['agglomerative_k2_silhouette']):.4f}`). "
                "Taken together, these methods suggest meaningful segmentation rather than random dispersion."
            ),
            (
                f"Anomaly detection flags `{int(metrics['anomaly_count'])}` institutions "
                f"(`{_format_pct(float(metrics['anomaly_share']))}`) as structurally unusual under Isolation Forest. "
                "This tail should be monitored as potential emerging stress or atypical balance-sheet behavior."
            ),
            "",
            "## Strongest Feature Relationships",
            *pair_lines,
            "",
            "## Artifacts Used",
            f"- `{metrics_path}`",
            f"- `{selected_run_dir / 'synopsis_pca_components.csv'}`",
            f"- `{selected_run_dir / 'synopsis_anomalies.csv'}`",
            f"- `{correlations_path}`",
            "",
            "## Next Actions",
            "- Re-run synopsis after each quarterly FDIC update to monitor structural drift.",
            "- Compare anomaly membership period-over-period to detect persistence vs transience.",
            "- Use top correlation pairs to guide feature refinement and hypothesis testing.",
            "",
        ]
    )
    report_path.write_text(text)
    return report_path

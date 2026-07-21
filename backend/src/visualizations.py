"""Render matplotlib figures and a markdown report from run artifacts.

Uses the non-interactive ``Agg`` backend so figures render headlessly (e.g. in CI).
Each chart builder returns ``None`` when its source artifact is missing, so partial
runs degrade gracefully into the report's "skipped charts" list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class VisualizationOutput:
    """Result of a visualization run: source run, figure paths, skips, and report."""

    run_dir: Path
    figure_paths: list[Path]
    skipped_charts: list[str]
    report_path: Path


def _latest_artifact_run(output_dir: Path) -> Path:
    candidates = sorted(path for path in output_dir.glob("*_*") if path.is_dir())
    for candidate in reversed(candidates):
        if any(
            (candidate / name).exists()
            for name in (
                "clustered_features.csv",
                "cluster_summary.csv",
                "synopsis_metrics.json",
                "synopsis_correlations.csv",
                "synopsis_anomalies.csv",
            )
        ):
            return candidate
    raise FileNotFoundError(f"No artifact runs with known visualization files found under {output_dir}")


def _save_permutation_importance_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    """Bar chart of the supervised model's permutation importances (top features)."""
    perm_path = run_dir / "supervised_permutation_importance.csv"
    if not perm_path.exists():
        return None

    perm_df = pd.read_csv(perm_path)
    if "feature" not in perm_df.columns or "permutation_importance_mean" not in perm_df.columns:
        return None
    if perm_df.empty:
        return None

    top = perm_df.sort_values("permutation_importance_mean", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    errors = top["permutation_importance_std"] if "permutation_importance_std" in top.columns else None
    ax.barh(top["feature"], top["permutation_importance_mean"], xerr=errors, color="#4e79a7")
    ax.set_title("Supervised Feature Importance (permutation)")
    ax.set_xlabel("Mean importance (drop in score when shuffled)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "permutation_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_cluster_map_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    """Scatter institutions on their first two PCA axes, colored by risk cluster.

    Needs both the synopsis PCA components and the clustering labels (i.e. a ``both``
    run); returns ``None`` if either artifact is missing. This 2-D "map" is the most
    intuitive single view of how the segments occupy the risk space.
    """
    pca_path = run_dir / "synopsis_pca_components.csv"
    clustered_path = run_dir / "clustered_features.csv"
    if not pca_path.exists() or not clustered_path.exists():
        return None

    pca_df = pd.read_csv(pca_path, index_col=0)
    clustered_df = pd.read_csv(clustered_path, index_col=0)
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        return None
    if "Risk_Cluster" not in clustered_df.columns:
        return None

    joined = pca_df[["PC1", "PC2"]].join(clustered_df["Risk_Cluster"], how="inner").dropna()
    if joined.empty:
        return None

    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (label, group) in enumerate(joined.groupby("Risk_Cluster")):
        ax.scatter(
            group["PC1"], group["PC2"], s=12, alpha=0.6,
            color=palette[idx % len(palette)], label=str(label),
        )
    ax.set_title("Risk Segments in PCA Space (PC1 vs PC2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(title="Risk Cluster", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    out_path = figures_dir / "cluster_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_cluster_size_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    clustered_path = run_dir / "clustered_features.csv"
    if not clustered_path.exists():
        return None

    clustered_df = pd.read_csv(clustered_path)
    if "Risk_Cluster" not in clustered_df.columns:
        return None

    counts = clustered_df["Risk_Cluster"].value_counts().sort_index()
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", color="#4e79a7", ax=ax)
    ax.set_title("Institution Count by Risk Cluster")
    ax.set_xlabel("Risk Cluster")
    ax.set_ylabel("Institutions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "cluster_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_cluster_feature_means_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    summary_path = run_dir / "cluster_summary.csv"
    if not summary_path.exists():
        return None

    summary_df = pd.read_csv(summary_path, index_col=0)
    if summary_df.empty:
        return None

    # Z-score each feature (column) across clusters so features on different scales
    # (fraction ratios vs. percentages) are visually comparable: color now shows how
    # high or low each cluster is on a feature relative to the other clusters.
    std = summary_df.std(axis=0).replace(0, 1.0)
    normalized = (summary_df - summary_df.mean(axis=0)) / std
    limit = float(max(abs(normalized.to_numpy().min()), abs(normalized.to_numpy().max()), 1.0))

    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(summary_df.columns)), 4.5))
    image = ax.imshow(normalized.values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("Cluster Feature Profile (z-scored across clusters)")
    ax.set_xticks(range(len(summary_df.columns)))
    ax.set_xticklabels(summary_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(summary_df.index)))
    ax.set_yticklabels(summary_df.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Std. devs from cross-cluster mean")
    fig.tight_layout()
    out_path = figures_dir / "cluster_feature_means.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_top_correlations_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    corr_path = run_dir / "synopsis_correlations.csv"
    if not corr_path.exists():
        return None

    corr_df = pd.read_csv(corr_path)
    required_columns = {"feature_left", "feature_right", "correlation"}
    if not required_columns.issubset(corr_df.columns):
        return None
    if corr_df.empty:
        return None

    top = corr_df.head(10).copy()
    top["pair"] = top["feature_left"] + " vs " + top["feature_right"]
    top = top.iloc[::-1]
    colors = top["correlation"].apply(lambda value: "#59a14f" if value >= 0 else "#e15759")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(top["pair"], top["correlation"], color=list(colors))
    ax.set_title("Top Feature Correlations")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature Pair")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "top_correlations.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_pca_variance_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    metrics_path = run_dir / "synopsis_metrics.json"
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    if "pca_explained_variance_pc1" not in metrics or "pca_explained_variance_top3" not in metrics:
        return None

    pc1 = float(metrics["pca_explained_variance_pc1"])
    top3 = float(metrics["pca_explained_variance_top3"])
    remaining = max(0.0, 1.0 - top3)

    labels = ["PC1", "Top 3 PCs", "Remaining"]
    values = [pc1, top3, remaining]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#bab0ab"])
    ax.set_title("PCA Explained Variance Overview")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance Share")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path = figures_dir / "pca_variance_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_anomaly_share_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    metrics_path = run_dir / "synopsis_metrics.json"
    anomalies_path = run_dir / "synopsis_anomalies.csv"
    if not metrics_path.exists() and not anomalies_path.exists():
        return None

    anomalies = 0
    normal = 0
    if anomalies_path.exists():
        anomaly_df = pd.read_csv(anomalies_path)
        if "anomaly_flag" in anomaly_df.columns:
            anomalies = int((anomaly_df["anomaly_flag"] == -1).sum())
            normal = int((anomaly_df["anomaly_flag"] != -1).sum())

    if metrics_path.exists() and (anomalies + normal == 0):
        metrics = json.loads(metrics_path.read_text())
        if "anomaly_count" in metrics and "rows_analyzed" in metrics:
            anomalies = int(metrics["anomaly_count"])
            rows = int(metrics["rows_analyzed"])
            normal = max(0, rows - anomalies)

    total = anomalies + normal
    if total == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Normal", "Anomalies"], [normal, anomalies], color=["#4e79a7", "#e15759"])
    ax.set_title("Anomaly Detection Breakdown")
    ax.set_ylabel("Institutions")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        1,
        anomalies + max(total * 0.02, 1),
        f"{(anomalies / total) * 100:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = figures_dir / "anomaly_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_visualization_report(
    run_dir: Path,
    report_dir: Path,
    figure_paths: list[Path],
    skipped_charts: list[str],
) -> Path:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    report_path = report_dir / f"VISUALIZATION_REPORT_{stamp}.md"
    figure_lines = []
    for figure in figure_paths:
        relative_path = figure.relative_to(report_dir)
        title = figure.stem.replace("_", " ").title()
        figure_lines.extend([f"### {title}", f"![{title}]({relative_path.as_posix()})", ""])

    skipped_lines = skipped_charts if skipped_charts else ["None"]

    text = "\n".join(
        [
            "# Visualization Report",
            "",
            "## Run Source",
            f"- `{run_dir}`",
            "",
            "## Generated Figures",
            *(figure_lines if figure_lines else ["No figures were generated.", ""]),
            "## Skipped Charts",
            *[f"- {item}" for item in skipped_lines],
            "",
        ]
    )
    report_path.write_text(text)
    return report_path


def generate_visualizations(
    output_dir: Path = Path("backend/artifacts"),
    report_dir: Path = Path("backend/artifacts/reports/secondary"),
    figures_dir: Path = Path("backend/artifacts/reports/secondary/figures"),
    run_dir: Path | None = None,
) -> VisualizationOutput:
    """Build all charts for a run and write a markdown report referencing them.

    Uses ``run_dir`` when given, else the latest artifact run under ``output_dir``.
    Charts whose source artifacts are absent are recorded as skipped rather than
    raising, and the populated ``VisualizationOutput`` is returned.
    """
    selected_run_dir = run_dir if run_dir is not None else _latest_artifact_run(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    skipped_charts: list[str] = []

    chart_builders = [
        ("cluster_map", _save_cluster_map_chart),
        ("permutation_importance", _save_permutation_importance_chart),
        ("cluster_size", _save_cluster_size_chart),
        ("cluster_feature_means", _save_cluster_feature_means_chart),
        ("top_correlations", _save_top_correlations_chart),
        ("pca_variance_overview", _save_pca_variance_chart),
        ("anomaly_breakdown", _save_anomaly_share_chart),
    ]

    for chart_name, builder in chart_builders:
        generated = builder(selected_run_dir, figures_dir)
        if generated is None:
            skipped_charts.append(f"{chart_name}: missing or invalid source artifact")
        else:
            figure_paths.append(generated)

    report_path = _write_visualization_report(
        run_dir=selected_run_dir,
        report_dir=report_dir,
        figure_paths=figure_paths,
        skipped_charts=skipped_charts,
    )
    return VisualizationOutput(
        run_dir=selected_run_dir,
        figure_paths=figure_paths,
        skipped_charts=skipped_charts,
        report_path=report_path,
    )
